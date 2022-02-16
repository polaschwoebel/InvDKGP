""" 
Note to the Reader:

In order to handle the generic spatial data we process features as a tuple: (coordinates,values,mask)
with shapes (bs,n,d), (bs,n,c), (bs,n) where bs is batchsize, n is the maximum number of points in the minibatch,
d is the dimension of the coordinates, and c is the number of channels of the feature map.
The mask specifies which elements are valid and ~mask specifies the elements that have been added through padding.
For the PointConv operation and networks elements are passed through the network as this tuple (coordinates,values,mask).

Naively for LieConv we would process (lie_algebra_elems,values,mask) with the same shapes but with d as the dimension of the group.
However, as an optimization to avoid repeated calculation of the pairs log(v^{-1}u)=log(e^{-b}e^{a}), we instead compute this for all
pairs once at the lifting stage which has the name 'ab_pairs' in the code and shape (bs,n,n,d). Subsampling operates on this matrix
by subsampling both n axes. abq_pairs also includes the q pairs (embedded orbit identifiers) so abq_pairs =  [log(e^{-b}e^{a}),qa,qb].
So the tuple  (abq_pairs,values,mask) with shapes (bs,n,n,d) (bs,n,c) (bs,n) is passed through the network. 
The 'Pass' module is used extensively to operate on only one of these and return the tuple with the rest unchanged, 
such as for computing a swish nonlinearity on values.
"""

#Pola's notes:
# - Linear layer --> Dense layer in tf, 
# got rid of all input dims in this type of layer since that's required in tf/keras; hope this still works

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
from .lie_conv_functions.utils import Expression, export, Named, Pass
from .lie_conv_functions.utils import FarthestSubsample, knn_point, index_points 
from .lie_conv_functions.lieGroups import Trivial, norm  # --> need only port trivial group
from .lie_conv_functions.masked_batchnorm import MaskBatchNormNd
# from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer

@export
def Swish():
    return Expression(lambda x: x*tf.math.sigmoid(x))


def LinearBNact(chin, chout, act='swish', bn=True):
    """assumes that the inputs to the net are shape (bs,n,mc_samples,c)"""
    assert act in ('relu', 'swish'), f"unknown activation type {act}"
    normlayer = MaskBatchNormNd(chout)
    # return Sequential([
    return [Pass(layers.Dense(chout),dim=1),
        # normlayer if bn else Sequential(),
        Pass(Swish() if act=='swish' else layers.ReLU(),dim=1)]


def WeightNet(in_dim, out_dim,act, bn,k=32):
    return Sequential([
        *LinearBNact(in_dim, k, act, bn),
        *LinearBNact(k, k, act, bn),
        *LinearBNact(k, out_dim, act, bn)])


class PointConv(layers.Layer):
    def __init__(self,chin,chout,mc_samples=32,xyz_dim=3,ds_frac=1,knn_channels=None,act='swish',bn=False,mean=False):
        super().__init__()
        self.chin = chin # input channels
        self.cmco_ci = 16 # a hyperparameter controlling size and bottleneck compute cost of weightnet
        self.xyz_dim = xyz_dim # dimension of the space on which convolution operates
        self.knn_channels = knn_channels # number of xyz dims on which to compute knn
        self.mc_samples = mc_samples # number of samples to use to estimate convolution
        self.weightnet = WeightNet(xyz_dim, self.cmco_ci, act, bn) # MLP - final layer to compute kernel vals (see A1)
        self.linear = layers.Dense(chout)        # final linear layer to compute kernel vals (see A1)
        self.mean=mean  # Whether or not to divide by the number of mc_samples
        assert ds_frac==1, "test branch no ds, will need to carefully check that subsample respects padding"
        self.subsample = FarthestSubsample(ds_frac,knn_channels=knn_channels) # Module for subsampling if ds_frac<1

    def extract_neighborhood(self,inp,query_xyz):
        """ inputs shape ([inp_xyz (bs,n,d)], [inp_vals (bs,n,c)], [query_xyz (bs,m,d)])"""
        inp_xyz,inp_vals,mask = inp
        neighbor_idx = knn_point(min(self.mc_samples, inp_xyz.shape[1]),
                    inp_xyz[:,:,:self.knn_channels], query_xyz[:,:,:self.knn_channels],mask)
        neighbor_xyz = index_points(inp_xyz, neighbor_idx) # (bs,n,mc_samples,d)
        neighbor_values = index_points(inp_vals, neighbor_idx) #(bs,n,mc_samples,c)
        neighbor_mask = index_points(mask,neighbor_idx) # (bs,n,mc_samples)
        return neighbor_xyz, neighbor_values, neighbor_mask

    def point_convolve(self,embedded_group_elems,nbhd_vals, nbhd_mask):
        """ embedded_group_elems: (bs,m,nbhd,d)
            nbhd_vals: (bs,m,mc_samples,ci)
            nbhd_mask: (bs,m,mc_samples)"""
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,mc_samples,d) -> (bs,m,mc_samples,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = tf.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,tf.zeros_like(penult_kernel_weights))
        nbhd_vals_m = tf.where(nbhd_mask.unsqueeze(-1),nbhd_vals,tf.zeros_like(nbhd_vals))
        #      (bs,m,mc_samples,ci) -> (bs,m,ci,mc_samples) @ (bs, m, mc_samples, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1)
        return convolved_vals

    def get_embedded_group_elems(self,output_xyz,nbhd_xyz):
        return output_xyz - nbhd_xyz
    
    def forward(self, inp):
        """inputs, and outputs have shape ([xyz (bs,n,d)], [vals (bs,n,c)])
            query_xyz has shape (bs,n,d)"""
        query_xyz, sub_vals, sub_mask = self.subsample(inp)
        nbhd_xyz, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_xyz)
        deltas = self.get_embedded_group_elems(query_xyz.unsqueeze(2), nbhd_xyz)
        convolved_vals = self.point_convolve(deltas, nbhd_vals, nbhd_mask)
        convolved_wzeros = tf.where(sub_mask.unsqueeze(-1),convolved_vals,tf.zeros_like(convolved_vals))
        return query_xyz, convolved_wzeros, sub_mask

def FPSindices(dists,frac,mask):
    """ inputs: pairwise distances DISTS (bs,n,n), downsample_frac (float), valid atom mask (bs,n) 
        outputs: chosen_indices (bs,m) """
    m = int(np.round(frac*dists.shape[1]))
    # device = dists.device
    bs,n = dists.shape[:2]
    chosen_indices = tf.zeros([bs, m], dtype=tf.int64)
    distances = tf.ones(bs, n) * 1e8
    a = tf.random.uniform(shape=(bs, ), minval=0, maxval=n, dtype=tf.int64) #choose random start
    idx = a%mask.sum(-1) + tf.concat([tf.zeros(1).long(),tf.math.cumsum(mask.sum(-1),axis=0)[:-1]],axis=0)
    farthest = tf.where(mask)[1][idx]
    B = tf.keras.backend.arange(bs, dtype=tf.int64)
    for i in range(m):
        chosen_indices[:, i] = farthest # add point that is farthest to chosen
        dist = tf.where(mask,dists[B,farthest],-100*tf.ones_like(distances)) # (bs,n) compute distance from new point to all others
        closer = dist < distances      # if dist from new point is smaller than chosen points so far
        distances[closer] = dist[closer] # update the chosen set's distance to all other points
        farthest = tf.reduce_max(distances, -1)[1] # select the point that is farthest from the set
    return chosen_indices


class FPSsubsample(tf.Module):
    def __init__(self,ds_frac,cache=False,group=None, batch_size=64):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache = cache
        self.cached_indices = None
        self.group = group
        self.batch_size = batch_size

    def forward(self, inp, withquery=False):
        abq_pairs,vals,mask = inp
        dist = self.group.distance if self.group else lambda ab: norm(ab,dim=-1)
        if self.ds_frac!=1:
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = FPSindices(dist(abq_pairs),self.ds_frac,mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = FPSindices(dist(abq_pairs),self.ds_frac,mask, batch_size=self.batch_size)
            B = tf.keras.backend.arange(query_idx.shape[0]).long()[:,None]
            subsampled_abq_pairs = abq_pairs[B,query_idx][B,:,query_idx]
            subsampled_values = vals[B,query_idx]
            subsampled_mask = mask[B,query_idx]
        else:
            subsampled_abq_pairs = abq_pairs
            subsampled_values = vals
            subsampled_mask = mask
            query_idx = None
        if withquery: return (subsampled_abq_pairs,subsampled_values,subsampled_mask, query_idx)
        return (subsampled_abq_pairs,subsampled_values,subsampled_mask)

class LieConv(PointConv):
    def __init__(self, *args, group=Trivial(), ds_frac=1, fill=1/3, cache=False, batch_size=64, 
                 knn=False, **kwargs):
        kwargs.pop('xyz_dim', None)
        super().__init__(*args, xyz_dim=group.lie_dim+2*group.q_dim, **kwargs)
        self.group = group  # Equivariance group for LieConv
        self.batch_size = batch_size
        # self.register_buffer('r',tf.constant(2.)) # Internal variable for local_neighborhood radius, set by fill
        self.r = tf.constant(2.) # Pola: not sure about this
        self.fill_frac = min(fill,1.) # Average Fraction of the input which enters into local_neighborhood, determines r
        self.knn = knn            # Whether or not to use the k nearest points instead of random samples for conv estimator
        self.subsample = FPSsubsample(ds_frac,cache=cache,group=self.group, batch_size=self.batch_size)
        self.coeff = .5  # Internal coefficient used for updating r
        self.fill_frac_ema = fill # Keeps track of average fill frac, used for logging only
        
    def extract_neighborhood(self,inp,query_indices):
        """ inputs: [pairs_abq (bs,n,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_abq (bs,m,mc_samples,d), neighbor_vals (bs,m,mc_samples,c)]"""

        # Subsample pairs_ab, inp_vals, mask to the query_indices
        pairs_abq, inp_vals, mask = inp
        if query_indices is not None:
            B = tf.keras.backend.arange(inp_vals.shape[0]).long()[:,None]
            abq_at_query = pairs_abq[B,query_indices]
            mask_at_query = mask[B,query_indices]
        else:
            abq_at_query = pairs_abq
            mask_at_query = mask
        vals_at_query = inp_vals
        dists = self.group.distance(abq_at_query) #(bs,m,n,d) -> (bs,m,n)
        dists = tf.where(mask[:,None,:].expand(*dists.shape),dists,1e8*tf.ones_like(dists))
        k = min(self.mc_samples,inp_vals.shape[1])
        
        # Determine ids (and mask) for points sampled within neighborhood (A4)
        if self.knn: # NBHD: KNN
            nbhd_idx = tf.math.top_k(dists,k,dim=-1,largest=False,sorted=False)[1] #(bs,m,nbhd)
            valid_within_ball = (nbhd_idx>-1)&mask[:,None,:]&mask_at_query[:,:,None]
            assert not tf.math.reduce_any(nbhd_idx>dists.shape[-1]), f"error with topk,\
                        nbhd{k} nans|inf{tf.math.reduce_any(tf.math.is_nan(dists)|tf.math.is_inf(dists))}"
        else: # NBHD: Sampled Distance Ball
            bs,m,n = dists.shape
            within_ball = (dists < self.r)&mask[:,None,:]&mask_at_query[:,:,None] # (bs,m,n)
            B = tf.keras.backend.arange(bs)[:,None,None]
            M = tf.keras.backend.arange(m)[None,:,None]
            noise = tf.zeros(bs,m,n)
            noise.uniform_(0,1)
            valid_within_ball, nbhd_idx =tf.math.top_k(within_ball+noise,k,dim=-1,largest=True,sorted=False)
            valid_within_ball = (valid_within_ball>1)
        
        # Retrieve abq_pairs, values, and mask at the nbhd locations
        B = tf.keras.backend.arange(inp_vals.shape[0]).long()[:,None,None].expand(*nbhd_idx.shape)
        M = tf.keras.backend.arange(abq_at_query.shape[1]).long()[None,:,None].expand(*nbhd_idx.shape)
        nbhd_abq = abq_at_query[B,M,nbhd_idx]     #(bs,m,n,d) -> (bs,m,mc_samples,d)
        nbhd_vals = vals_at_query[B,nbhd_idx]   #(bs,n,c) -> (bs,m,mc_samples,c)
        nbhd_mask = mask[B,nbhd_idx]            #(bs,n) -> (bs,m,mc_samples)
        
        if self.training and not self.knn: # update ball radius to match fraction fill_frac inside
            navg = tf.sum(tf.sum((within_ball.float()), -1))/ tf.sum(mask_at_query[:,:,None])
            avg_fill = (navg/mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff*(self.fill_frac - avg_fill)
            self.fill_frac_ema += .1*(avg_fill-self.fill_frac_ema)
        return nbhd_abq, nbhd_vals, (nbhd_mask&valid_within_ball.bool())

    # def log_data(self,logger,step,name):
    #     logger.add_scalars('info', {f'{name}_fill':self.fill_frac_ema}, step=step)
    #     logger.add_scalars('info', {f'{name}_R':self.r}, step=step)

    def point_convolve(self,embedded_group_elems,nbhd_vals,nbhd_mask):
        """ Uses generalized PointConv trick (A1) to compute convolution using pairwise elems (aij) and nbhd vals (vi).
            inputs [embedded_group_elems (bs,m,mc_samples,d), nbhd_vals (bs,m,mc_samples,ci), nbhd_mask (bs,m,mc_samples)]
            outputs [convolved_vals (bs,m,co)]"""
        bs, m, nbhd, ci = nbhd_vals.shape  # (bs,m,mc_samples,d) -> (bs,m,mc_samples,cm*co/ci)
        _, penult_kernel_weights, _ = self.weightnet((None,embedded_group_elems,nbhd_mask))
        penult_kernel_weights_m = tf.where(nbhd_mask.unsqueeze(-1),penult_kernel_weights,tf.zeros_like(penult_kernel_weights))
        nbhd_vals_m = tf.where(nbhd_mask.unsqueeze(-1),nbhd_vals,tf.zeros_like(nbhd_vals))
        #      (bs,m,mc_samples,ci) -> (bs,m,ci,mc_samples) @ (bs, m, mc_samples, cmco/ci) -> (bs,m,ci,cmco/ci) -> (bs,m, cmco) 
        partial_convolved_vals = (nbhd_vals_m.transpose(-1,-2)@penult_kernel_weights_m).view(bs, m, -1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,m,cmco) -> (bs,m,co)
        if self.mean: convolved_vals /= nbhd_mask.sum(-1,keepdim=True).clamp(min=1) # Divide by num points
        return convolved_vals

    def forward(self, inp):
        """inputs: [pairs_abq (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs [subsampled_abq (bs,m,m,d)], [convolved_vals (bs,m,co)]"""
        sub_abq, sub_vals, sub_mask, query_indices = self.subsample(inp,withquery=True)
        nbhd_abq, nbhd_vals, nbhd_mask = self.extract_neighborhood(inp, query_indices)
        convolved_vals = self.point_convolve(nbhd_abq, nbhd_vals, nbhd_mask)
        convolved_wzeros = tf.where(sub_mask.unsqueeze(-1),convolved_vals,tf.zeros_like(convolved_vals))
        return sub_abq, convolved_wzeros, sub_mask



# @export
# def pConvBNrelu(in_channels,out_channels,bn=True,act='swish',**kwargs):
#     return Sequential([
#         PointConv(in_channels,out_channels,bn=bn,**kwargs),
#         # MaskBatchNormNd(out_channels) if bn else Sequential(), # TODO: implement batchnorm
#         Pass(Swish() if act=='swish' else layers.ReLU(),dim=1)
#     ])

# @export
# def LieConvBNrelu(in_channels,out_channels,bn=True,act='swish', batch_size=64, **kwargs):
#     return Sequential([
#         LieConv(in_channels,out_channels,bn=bn, batch_size=batch_size, **kwargs),
#         MaskBatchNormNd(out_channels) if bn else Sequential(),
#         Pass(Swish() if act=='swish' else layers.ReLU(),dim=1)
#     ])


# class BottleBlock(layers.Layer):
#     """ A bottleneck residual block as described in figure 5"""
#     def __init__(self,chin,chout,conv,bn=False,act='swish',fill=None):
#         super().__init__()
#         assert chin<= chout, f"unsupported channels chin{chin}, chout{chout}. No upsampling atm."
#         nonlinearity = Swish if act=='swish' else layers.ReLU
#         self.conv = conv(chin//4,chout//4,fill=fill) if fill is not None else conv(chin//4,chout//4)
#         self.net = Sequential([
#             MaskBatchNormNd(chin) if bn else Sequential(),
#             Pass(nonlinearity(),dim=1),
#             Pass(layers.Dense(chin//4),dim=1),
#             MaskBatchNormNd(chin//4) if bn else Sequential(),
#             Pass(nonlinearity(),dim=1),
#             self.conv,
#             MaskBatchNormNd(chout//4) if bn else Sequential(),
#             Pass(nonlinearity(),dim=1),
#             Pass(layers.Dense(chout),dim=1),
#         ])
#         self.chin = chin

#     def forward(self,inp):
#         sub_coords, sub_values, mask = self.conv.subsample(inp)
#         new_coords, new_values, mask = self.net(inp)
#         new_values[...,:self.chin] += sub_values
#         return new_coords, new_values, mask

def bottle_block(inp, chin,chout,conv, fill=None, bn=False, act='swish', batch_size=64):
    conv = conv(chin//4,chout//4,fill=fill) if fill is not None else conv(chin//4,chout//4)
    nonlinearity = Swish if act=='swish' else layers.ReLU

    def net(inp, conv):
        # sub_coords, sub_values, mask = MaskBatchNormNd(chin)(inp) if bn else Sequential()(inp)
        sub_coords, sub_values, mask = inp
        sub_values = nonlinearity()(sub_values)
        sub_values = layers.Dense(chin//4)(sub_values)
        #sub_coords, sub_values, mask = MaskBatchNormNd(chin)([sub_coords, sub_values, mask]) if bn else Sequential()([sub_coords, sub_values, mask])
        sub_values = nonlinearity()(sub_values)
        sub_coords, sub_values, mask = conv([sub_coords, sub_values, mask])
        #[sub_coords, sub_values, mask] = MaskBatchNormNd(chout//4)([sub_coords, sub_values, mask]) if bn else Sequential()([sub_coords, sub_values, mask])
        sub_values = nonlinearity()(sub_values)
        sub_values = layers.Dense(chout)(sub_values)
        return [sub_coords, sub_values, mask]

    sub_coords, sub_values, mask = conv.subsample.forward(inp)
    new_coords, new_values, mask = net(inp, conv)
    # print(new_values.shape)
    # print(new_values[...,:chin].shape)
    # print(sub_values.shape)
    # exit()
    # new_values[...,:chin] += sub_values  Pola: TODO! This  seems to have gone wrong with the dims
    new_values += sub_values
    return new_coords, new_values, mask



# class GlobalPool(layers.Layer):
# """computes values reduced over all spatial locations (& group elements) in the mask"""
# def __init__(self,mean=False):
#     super().__init__()
#     self.mean = mean
    
def global_pool(x, mean=False):
    """x [xyz (bs,n,d), vals (bs,n,c), mask (bs,n)]"""
    print('length is', len(x))
    if len(x) == 2: return x[1].mean(1)
    coords, vals,mask = x
    print('vals:', vals.shape)
    print('mask -- is this correct?', mask.shape) # Is the mask shape correct??
    summed = tf.reduce_sum(vals[tf.tile(mask, [1, 1, 6])]) # Pola: is this equiv with below?
    # summed = tf.where(mask.unsqueeze(-1),vals,tf.zeros_like(vals)).sum(1)
    if mean:
        summed /= mask.sum(-1).unsqueeze(-1)
    return summed

@export
class MolecLieResNet(Layer, metaclass=Named): #tf.keras.Sequential, 
    """ Generic LieConv architecture from Fig 5. Relevant Arguments:
        [Fill] specifies the fraction of the input which is included in local neighborhood. 
                (can be array to specify a different value for each layer)
        [nbhd] number of samples to use for Monte Carlo estimation (p)
        [chin] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
        [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
        [num_layers] number of BottleNeck Block layers in the network
        [k] channel width for the network. Can be int (same for all) or array to specify individually.
        [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
        [Group] Chosen group to be equivariant to.
        [bn] whether or not to use batch normalization. Recommended in all cases except dynamical systems.
        """
    def __init__(self, ds_frac=1, num_outputs=1, k=1536, nbhd=np.inf,
                act="swish", bn=True, num_layers=6, mean=True, per_point=True,pool=True,
                liftsamples=1, fill=1/4, group=Trivial(), charge_scale=None, knn=False,cache=False, batch_size=64, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        # exit()
        if isinstance(fill,(float,int)):
            fill = [fill]*num_layers
        if isinstance(k,int):
            k = [k]*(num_layers+1)
        conv = lambda ki,ko,fill: LieConv(ki, ko, mc_samples=nbhd,
                                          ds_frac=ds_frac, bn=bn, act=act, mean=mean, batch_size=self.batch_size,
                                          group=group,fill=fill,cache=cache,knn=knn, **kwargs)
        self.fill = fill
        self.k = k
        self.bn = bn
        self.act = act
        self.pool = pool
        self.conv = conv
        self.mean = mean
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.liftsamples = liftsamples
        self.per_point=per_point
        self.group = group
        self.charge_scale = charge_scale

        def featurize(atomic_coords, charges):
            #one_hot_charges = (mb['one_hot'][:,:,:,None]*c_vec[:,:,None,:]).float().reshape(*charges.shape,-1) # our charges are already one hot
            atomic_coords = tf.cast(atomic_coords, tf.float64)
            # charges = mb[1] / self.charge_scale
            # c_vec = tf.stack([tf.ones_like(charges),charges,charges**2],axis=-1) # 
            # one_hot_charges = (mb['one_hot'][:,:,:,None]*c_vec[:,:,None,:]).float().reshape(*charges.shape,-1)
            
            # I am just using plain one-hot encoded charges here, that's probably not quite right... 
            atom_mask = charges > 0
            # print('atomic_coords:', atomic_coords.shape, 'charges:', charges.shape, 'atom_mask:', atom_mask.shape)
            # exit()
            return atomic_coords, charges, atom_mask

        atomic_coords = Input(shape=(29, 3), name='atomic_coords')
        charges = Input(shape=(29, 6), name='charges')
        # atom_mask = Input(29)
        
        # with tf.stop_gradient(atomic_coords, charges): # Pola: not sure about this line? 
        atomic_coords, charges, atom_mask = featurize(atomic_coords, charges)
        lifted_atomic_coords, charges, atom_mask = self.group.lift([atomic_coords, charges, atom_mask], self.liftsamples)
        charges = layers.Dense(self.k[0], input_shape=(29, 6))(charges)    #embedding layer, TODO: don't have this hardcoded 
        for i in range(self.num_layers):
            atomic_coords, charges, atom_mask = bottle_block([atomic_coords, charges, atom_mask], 
                self.k[i], self.k[i+1], self.conv, bn=self.bn,act=self.act,fill=self.fill[i], batch_size=batch_size)
        # atomic_coords, charges, atom_mask = MaskBatchNormNd(self.k[-1])([atomic_coords, charges, atom_mask]) if self.bn else Sequential()([atomic_coords, charges, atom_mask]) # Pola: removed input to BAtch Norm self.k[-1]
        charges = Swish()(charges) if self.act=='swish' else layers.ReLU()(charges)
        charges = layers.Dense(self.num_outputs)(charges)
        pooled = global_pool([atomic_coords, charges, atom_mask], mean=self.mean) if self.pool else Expression(lambda x: x[1])([atomic_coords, charges, atom_mask])
        pooled = pooled.squeeze(-1)
        return Model(inputs=[atomic_coords, charges], outputs=pooled, name='LieResNet')


# @export
# class MolecLieResNet(LieResNet):
#     def __init__(self, num_species, charge_scale, group=Trivial(), **kwargs):
#         super().__init__(chin=3*num_species,num_outputs=1,group=group,ds_frac=1, **kwargs)
#         self.charge_scale = charge_scale
        # note: gotten rid of DA (aug input) here 


    # def forward(self,mb):
    #     print('mb:', mb)
    #     # with torch.no_grad(): # Pola: not sure about this line? 
    #     with tf.stop_gradient(mb):  
    #         atomic_coords, charges, atom_mask = self.featurize(mb)
    #     return super().forward(atomic_coords, charges, atom_mask).squeeze(-1)







