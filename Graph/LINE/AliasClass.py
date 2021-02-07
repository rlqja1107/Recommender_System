
class AliasSampling(object):
    def __init__(self, dist):
        self.dist=dist
        self.construct_alias_table()

    def construct_alias_table(self):
