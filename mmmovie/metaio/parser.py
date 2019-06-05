from mmcv.fileio import load, dump


class MetaParser(object):
    def __init__(self):
         super(MetaParser, self).__init__()
    
    def parse_metainfo(self):
        raise NotImplementedError
    
    def parse_cast_anotation(self):
        raise NotImplementedError

    def parse_synopsis(self):
        raise NotImplementedError
    
    def parse_scenario_boundary(self):
        raise NotImplementedError
