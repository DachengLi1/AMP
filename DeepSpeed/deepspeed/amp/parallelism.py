"""
Stores information for each dimension of parallelism
"""
class parallelism():
    def __init__(self, pp: int, dp: int, mp: int, pp_parts: list, rank_map: dict):
        
        self.pp = pp
        self.dp = dp
        self.mp = mp
        
        """
         The order of axis is (pp, dp, mp). For example, if dp=2,mp=3,pp=4,
         the numbe gpu at number 5 stands for (pp=0, dp=1, mp=2).

          
         An example pp_parts for pp = 2 is [0, 14, 30] for a load-balanced
         GPT2 model with num_layer = 24.

        """

        self._check_valid_rank(rank_map)
        self.rank_map = rank_map
        self.pp_parts = pp_parts

    """
    Check whether the rank_map is a valid permutation from 0 to num(gpu)-1.
    """
    def _check_valid_rank(self, rank_map):
        base = []
        for k, v in rank_map.items():
            base.extend(v)

        bijection = (len(rank_map) == self.pp * self.dp * self.mp)
        validity = (sorted(base) == list(range(len(base)))) 
        assert validity and bijection, "rank map is not a permutation from 0 to num_gpus-1."

    """
    Returns this as (rank_map, pp, dp, mp, pp_config)
    """
    def get_repr(self):
        return (self.rank_map, self.pp, self.dp, self.mp, self.pp_parts)


    """
    Returns the rank to axis. If pp=dp=mp=2, rank 3 gives (0,1,1)
    """
    def rank2axis(self, rank):
        pp = rank // (self.mp * self.dp)
        remainder = rank % (self.mp * self.dp)
        
        dp = remainder // (self.mp)
        remainder = remainder % self.mp
        
        mp = remainder

        returns (pp, dp, mp)

    """
    Returns the axis to rank. If pp=dp=mp=2, (0,1,1) gives 3
    """
    def axis2rank(self, axis):
        pp, dp, mp = axis
        return mp + self.mp * dp + (self.mp * self.dp) * pp


