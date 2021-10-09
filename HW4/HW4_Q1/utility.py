# Utility for printing information more friendly.
def format_decorator(func):
    def inner1(*args, **kwargs):
        func_name = func.__name__
        begin_title = "Beginning of " + func_name
        left_num_sharp = (80 - len(begin_title) - 2)>>1
        right_num_sharp = 80 - len(begin_title) - 2 - left_num_sharp
        print("\n\n{0} {1} {2}\n".format('#'*left_num_sharp,
                                         begin_title,
                                         '#'*right_num_sharp))
        ret = func(*args, **kwargs)

        end_title = "Ending of " + func_name
        left_num_sharp = (80 - len(end_title) - 2)>>1
        right_num_sharp = 80 - len(end_title) - 2 - left_num_sharp
        print("\n{0} {1} {2}".format('#'*left_num_sharp, end_title,
                                   '#'*right_num_sharp))

        return ret

    return inner1


def report_root_mean_squared_error(notation, pred, target):
    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(pred, target))
    print("Root mean squared errors on {}: {}".format(notation,
                                                      rmse))
