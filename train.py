import configparser


import config

def logging(file):
    def write_log(s):
        print(s,'')
        with open(file, 'a') as f:
            f.write(s)
    return write_log

def train(epoch):
    return

def eval(epoch):
    return

def main():
    for i in range(1, config.epoch+1):
        if not config.notrain:
            train(i)
        else:
            eval(i)
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
