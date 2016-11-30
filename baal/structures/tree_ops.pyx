from math import ceil, floor
from copy import deepcopy


### constants

cdef enum k_TreeOp: k_LeftOp, k_RightOp, k_SubOp

cpdef enum k_PointType: k_SubPoint, k_InsPoint

cdef enum k_PartOfSpeech:
    k_ROOT,
    k_NP, k_NNP, k_NN,
    k_VP, k_V, k_VBD, k_VBZ,
    k_ADVP, k_PP, k_ADJ

cdef _to_constant(str sym):
    return {'ROOT': k_ROOT, 'NP': k_NP, 'NNP': k_NNP,
             'VP': k_VP, 'V': k_V, 'VBD': k_VBD, 'VBZ': k_VBZ,
             'ADVP': k_ADVP, 'PP': k_PP, 'ADJ': k_ADJ,
             'LEFT': k_LeftOp, 'RIGHT': k_RightOp, 'UP': k_SubOp,
             'SUB': k_SubPoint, 'INS': k_InsPoint}[sym.upper()]

cdef _validate_pos(pos):
    if isinstance(pos, str):
        return _to_constant(pos)
    elif isinstance(pos, k_PartOfSpeech):
        return pos
    else:
        raise ValueError("wrong part of speech type")

cdef _validate_op(op_type):
    if isinstance(op_type, str):
        return _to_constant(op_type)
    elif isinstance(op_type, k_TreeOp):
        return op_type
    else:
        raise ValueError("Wrong kind of variable passed for tree op type <{}>".format(type(op_type)))

cdef _validate_point(point):
    if isinstance(point, str):
        return _to_constant(point)
    elif isinstance(point, k_PointType):
        return point
    else:
        raise ValueError("Wrong kind of variable passed for point type <{}>".format(type(point)))

cdef str k_NullHLF = "ginf"
cdef tuple k_NullGorn = (-1,)


cdef class AttachmentPoint(object):
    cdef public bint free
    cdef public tuple gorn
    cdef public tuple frontier
    cdef public float frontier_increment
    cdef public str hlf

    cdef k_PointType type

    ### unsure about strings

    def __cinit__(self):
        self.frontier = (0.0, 0.0)
        self.frontier_increment = 0.01
        self.hlf = k_NullHLF

    def __init__(self, bint free, k_PartOfSpeech pos, tuple gorn, k_PointType type):
        self.free = free
        self.gorn = gorn
        self.type = _validate_point(type)
        self.pos = _validate_pos(pos)

    @classmethod
    def root(cls):
        print(isinstance(k_SubPoint, k_PointType))
        print(type(k_LeftOp))
        return cls(1, k_ROOT, (0,), k_SubPoint)

    @classmethod
    def make(cls, str pos=None, tuple gorn=None, str pt_type=None):
        pos = _validate_pos(pos or k_ROOT)
        gorn = gorn or k_NullGorn
        pt_type = _validate_point(pt_type or k_SubPoint)
        return cls(1, pos, gorn, pt_type)

    def __deepcopy__(self, memo):
        result = AttachmentPoint(self.free, self.pos, self.gorn, self.type)
        memo[id(self)] = result
        result.frontier = self.frontier
        result.frontier_increment = self.frontier_increment
        result.hlf = self.hlf
        return result

    property left_insert:
        def __get__(self):
            self.frontier = (round(self.frontier[0] - self.frontier_increment, 2),
                             self.frontier[1])
            return self.frontier[0]

        def __set__(self, x):
            raise ValueError

    property right_insert:
        def __get__(self):
            self.frontier = (self.frontier[0],
                             round(self.frontier[1] + self.frontier_increment, 2))
            return self.frontier[1]

        def __set__(self, x):
            raise ValueError

    property left_sub:
        def __get__(self):
            self.frontier = (ceil(self.frontier[0]) - 1.0, self.frontier[1])
            return self.frontier[0]

        def __set__(self, x):
            raise ValueError

    property right_sub:
        def __get__(self):
            self.frontier = (self.frontier[0], floor(self.frontier[1]) + 1.0)
            return self.frontier[1]

        def __set__(self, x):
            raise ValueError

    cpdef match(self, op, verbose=0):
        pos_match = self.pos_symbol == op.target.pos_symbol
        gorn_match = (self.gorn == op.target.gorn or op.target.gorn == None)
        hlf_match = self.hlf == op.target.hlf
        #xor
        type_match = (((self.pt_type == k_InsPoint) and op.is_insertion)
                      or ((self.pt_type == k_SubPoint) and not op.is_insertion))

        if verbose:
            print("POS MATCH: {}".format(pos_match))
            print("GORN MATCH: {}".format(gorn_match))
            print("HLF MATCH: {}".format(hlf_match))
            print("TYPE MATCH: {}".format(type_match))
            print("--OP ATTACHMENT: {}".format(op.target.attach_direction))
        return self.free and pos_match and gorn_match and hlf_match and type_match

cdef class Target(object):
    cdef tuple gorn
    cdef str hlf
    cdef k_TreeOp op
    cdef k_PartOfSpeech pos
    cdef bint is_insertion

    def __init__(self, k_TreeOp op, k_PartOfSpeech pos, tuple gorn, str hlf):
        self.op = op
        self.pos = pos
        self.gorn = gorn
        self.hlf = hlf
        if self.op != k_SubOp:
            self.is_insertion = True
        else:
            self.is_insertion = False

cdef class AttachmentOperation(object):
    cdef public Target target

    def __cinit__(self, Target attach_target):
        self.target = attach_target

    @classmethod
    def make(cls, op_type, pos, tuple gorn=None, hlf=None):
        op = _validate_op(op_type)
        pos = _validate_pos(pos)
        gorn = gorn or k_NullGorn
        hlf = hlf or k_NullHLF
        return cls(Target(op, pos, gorn, hlf))

    property is_insertion:
        def __get__(self):
            return self.target.is_insertion

        def __set__(self, value):
            raise ValueError

