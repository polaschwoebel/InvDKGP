from lie_conv.utils import export, Named
import tensorflow as tf


@export
def norm(x, dim):
    return tf.math.sqrt(tf.reduce_sum((x**2), axis=dim))


class LieGroup(object,metaclass=Named):
    """ The abstract Lie Group requiring additional implementation of exp,log, and lifted_elems
        to use as a new group for LieConv. rep_dim,lie_dim,q_dim should additionally be specified."""
    rep_dim = NotImplemented  # dimension on which G acts. (e.g. 2 for SO(2))
    lie_dim = NotImplemented  # dimension of the lie algebra of G. (e.g. 1 for SO(2))
    q_dim = NotImplemented  # dimension which the quotient space X/G is embedded. (e.g. 1 for SO(2) acting on R2)

    def __init__(self, alpha=.2):
        super().__init__()
        self.alpha = alpha

    def exp(self, a):
        """ Computes (matrix) exponential Lie algebra elements (in a given basis).
            ie out = exp(\sum_i a_i A_i) where A_i are the exponential generators of G.
            Input: [a (*,lie_dim)] where * is arbitrarily shaped
            Output: [exp(a) (*,rep_dim,rep_dim)] returns the matrix for each."""
        raise NotImplementedError

    def log(self, u):
        """ Computes (matrix) logarithm for collection of matrices and converts to Lie algebra basis.
            Input [u (*,rep_dim,rep_dim)]
            Output [coeffs of log(u) in basis (*,d)] """
        raise NotImplementedError

    def lifted_elems(self, xyz, nsamples):
        """ Takes in coordinates xyz and lifts them to Lie algebra elements a (in basis)
            and embedded orbit identifiers q. For groups where lifting is multivalued
            specify nsamples>1 as number of lifts to do for each point.
            Inputs: [xyz (*,n,rep_dim)],[mask (*,n)], [mask (int)]
            Outputs: [a (*,n*nsamples,lie_dim)],[q (*,n*nsamples,q_dim)]"""
        raise NotImplementedError

    def inv(self, g):
        """ We can compute the inverse of elements g (*,rep_dim,rep_dim) as exp(-log(g))"""
        return self.exp(-self.log(g))

    def distance(self, abq_pairs):
        """ Compute distance of size (*) from [abq_pairs (*,lie_dim+2*q_dim)].
            Simply computes alpha*norm(log(v^{-1}u)) +(1-alpha)*norm(q_a-q_b),
            combined distance from group element distance and orbit distance."""
        ab_dist = norm(abq_pairs[...,:self.lie_dim],dim=-1)
        qa = abq_pairs[...,self.lie_dim:self.lie_dim+self.q_dim]
        qb = abq_pairs[...,self.lie_dim+self.q_dim:self.lie_dim+2*self.q_dim]
        qa_qb_dist = norm(qa-qb,dim=-1)
        return ab_dist*self.alpha + (1-self.alpha)*qa_qb_dist

    def lift(self, x, nsamples, **kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
            returns (a,v) with shapes [(*,n*nsamples,lie_dim),(*,n*nsamples,c)"""
        p,v,m = x
        expanded_a,expanded_q = self.lifted_elems(p,nsamples,**kwargs) # (bs,n*ns,d), (bs,n*ns,qq)
        # print(expanded_q)
        # exit()
        nsamples = expanded_a.shape[-2]//m.shape[-1]
        # expand v and mask like q
        expanded_v = v[...,None,:].repeat((1,)*len(v.shape[:-1])+(nsamples,1)) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(*expanded_a.shape[:-1],v.shape[-1]) # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[...,None].repeat((1,)*len(v.shape[:-1])+(nsamples,)) # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(*expanded_a.shape[:-1]) # (bs,n,ns) -> (bs,n*ns)
        # convert from elems to pairs
        paired_a = self.elems2pairs(expanded_a) #(bs,n*ns,d) -> (bs,n*ns,n*ns,d)
        if expanded_q is not None:
            q_in = tf.expand_dims(expanded_q, -2).expand(*paired_a.shape[:-1],1)
            q_out = tf.expand_dims(expanded_q, -3).expand(*paired_a.shape[:-1],1)
            embedded_locations = tf.concat([paired_a,q_in,q_out],axis=-1)
        else:
            embedded_locations = paired_a
        return (embedded_locations,expanded_v,expanded_mask)

    def expand_like(self,v,m,a):
        nsamples = a.shape[-2]//m.shape[-1]
        expanded_v = v[...,None,:].repeat((1,)*len(v.shape[:-1])+(nsamples,1)) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c)
        expanded_v = expanded_v.reshape(*a.shape[:2],v.shape[-1]) # (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = m[...,None].repeat((1,)*len(v.shape[:-1])+(nsamples,)) # (bs,n) -> (bs,n,ns)
        expanded_mask = expanded_mask.reshape(*a.shape[:2]) # (bs,n,ns) -> (bs,n*ns)
        return expanded_v, expanded_mask
    
    def elems2pairs(self,a):
        """ computes log(e^-b e^a) for all a b pairs along n dimension of input.
            inputs: [a (bs,n,d)] outputs: [pairs_ab (bs,n,n,d)] """
        vinv = self.exp(tf.expand_dims(-a, -3))
        u = self.exp(tf.expand_dims(a, -2))
        return self.log(vinv@u)    # ((bs,1,n,d) -> (bs,1,n,r,r))@((bs,n,1,d) -> (bs,n,1,r,r))

    def BCH(self,a,b,order=2):
        """ Baker Campbell Hausdorff formula"""
        assert order <= 4, "BCH only supported up to order 4"
        B = self.bracket
        z = a+b
        if order==1: return z
        ab = B(a,b)
        z += (1/2)*ab
        if order==2: return z
        aab = B(a,ab)
        bba = B(b,-ab)
        z += (1/12)*(aab+bba)
        if order==3: return z
        baab = B(b,aab)
        z += -(1/24)*baab
        return z
    
    def bracket(self,a,b):
        """Computes the lie bracket between a and b, assumes a,b expressed as vectors"""
        A = self.components2matrix(a)
        B = self.components2matrix(b)
        return self.matrix2components(A@B-B@A)

    def __str__(self):
        return f"{self.__class__}({self.alpha})" if self.alpha!=.2 else f"{self.__class__}"
    def __repr__(self):
        return str(self)

# @export
# def LieSubGroup(liegroup,generators):
#    
#     class subgroup(liegroup):
#        
#         def __init__(self,*args,**kwargs):
#             super().__init__(*args,**kwargs)
#             self.orig_dim = self.lie_dim
#             self.lie_dim = len(generators)
#             self.q_dim = self.orig_dim-len(generators)

#         def exp(self,a_small):
#             a_full = tf.zeros(*a_small.shape[:-1],self.orig_dim,
#                         device=a_small.device,dtype=a_small.dtype)
#             a_full[...,generators] = a_small
#             return super().exp(a_full)
#        
#         def log(self,U):
#             return super().log(U)[...,generators]
#        
#         def components2matrix(self,a_small):
#             a_full = tf.zeros(*a_small.shape[:-1],self.orig_dim,
#                          device=a_small.device,dtype=a_small.dtype)
#             a_full[...,generators] = a_small
#             return super().components2matrix(a_full)
#        
#         def matrix2components(self,A):
#             return super().matrix2components(A)[...,generators]
#         def lifted_elems(self,pt,nsamples=1):
#             """ pt (bs,n,D) mask (bs,n), per_point specifies whether to
#                 use a different group element per atom in the molecule"""
#             a_full,q = super().lifted_elems(pt,nsamples)
#             a_sub = a_full[...,generators]
#             complement_generators = list(set(range(self.orig_dim))-set(generators))
#             new_qs = a_full[...,complement_generators]
#             q_sub = tf.concat([q,new_qs],dim=-1) if q is not None else new_qs
#             return a_sub,q_sub
#         # def __str__(self):
#         #     return f"Subgroup({str(liegroup)},{generators})"
#     return subgroup


@export
class Trivial(LieGroup):
    lie_dim = 0
    def __init__(self,dim=2):
        super().__init__()
        self.q_dim = dim
        self.rep_dim = dim

    def lift(self, x, nsamples,**kwargs):
        assert nsamples == 1, "Abelian group, no need for nsamples"
        p, v, m = x
        bs, n, d = p.shape
        # print('bs:', bs, 'n:', n, 'd:', d)
        # qa = p[..., :, None, :].expand(bs,n,n,d)
        # qb = p[...,None,:,:].expand(bs,n,n,d)
        qa = tf.tile(p[..., :, None, :], (1,1,n,1))
        qb = tf.tile(p[...,None,:,:], (1,n,1,1))
        q = tf.concat([qa,qb],axis=-1)
        return q,v,m
    # def distance(self,abq_pairs):
    #     qa = abq_pairs[...,:self.q_dim]
    #     qb = abq_pairs[...,self.q_dim:]
    #     return norm(qa-qb,dim=-1)
