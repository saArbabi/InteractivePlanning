"""
state_col = ['vel', 'pc', 'act_long','act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'lc_type', 'exists', 'exists', 'exists']
"""
class StateIndxs():
    def __init__(self):
        self.set_stateIndex()

    def set_stateIndex(self):
        self.indx_m = {}
        self.indx_y = {}
        self.indx_f = {}
        self.indx_fadj = {}
        i = 0
        for name in ['vel', 'pc', 'act_long','act_lat']:
            self.indx_m[name] = i
            i += 1
        for name in ['vel', 'dx', 'act_long','act_lat']:
            self.indx_y[name] = i
            i += 1
        for name in ['vel', 'dx', 'act_long','act_lat']:
            self.indx_f[name] = i
            i += 1
        for name in ['vel', 'dx', 'act_long','act_lat']:
            self.indx_fadj[name] = i
            i += 1

        self.indx_acts = [
                [self.indx_m['act_long'], self.indx_m['act_lat']],
                [self.indx_y['act_long'], self.indx_y['act_lat']],
                [self.indx_f['act_long'], self.indx_f['act_lat']],
                [self.indx_fadj['act_long'], self.indx_fadj['act_lat']]]
