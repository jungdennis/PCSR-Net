def create_model_sr(scale=4):
    from .PCSR import model_sr as M

    m = M(scale)

    return m

def create_model_reid(database, base, fold):
    from .PCSR import model_reid as M

    m = M(database, base, fold)

    return m