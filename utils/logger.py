from distutils.log import Log
import neptune.new as neptune


class Logger:
    def __init__(self, cfg, is_neptune):
        self.is_neptune = is_neptune
        if is_neptune:
            self.nlogger = neptune.init(
                project='raeyo/UOSP',
                api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0OTA1Mzk4OS04MWI4LTQ5YjctYTViZi1iZDEyNjFlOWJmMzAifQ==')
            self.nlogger["parameters"] = cfg
        else:
            self.nlogger = None
    
        self.log = ""

    def logging(self, name, value):
        if self.is_neptune:
            self.nlogger[name].log(value)

        self.log += "{}: {:.6f}\n".format(name, value)

    def __str__(self):
        log = self.log
        self.log = ""
        return log



if __name__=="__main__":
    logger = Logger(cfg={}, is_neptune=False)

    for i in range(10):
        logger.logging("train/loss", 1)
    print(logger)
