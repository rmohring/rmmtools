import datetime

class Timer(object):
    def __init__(self):
        self.reset()

    def lap(self, text="now", flush=True):
        now = datetime.datetime.today()

        print(f"{now.strftime('%Y-%m-%d %H:%M:%S.%f')}: {text}", flush=flush)
        if self.SAVED_DTTMS:
            elapsed = (now - self.SAVED_DTTMS[-1][0]).total_seconds()
            print(f"-- {elapsed} elapsed since last lap", flush=flush)
        else:
            print(f"-- Starting first lap --", flush=flush)

        self.SAVED_DTTMS.append((now, text))

        return

    def reset(self):
        self.SAVED_DTTMS = []

    def output(self, msgs, filename=None):
        for msg in msgs:
            print(msg, flush=True)

        if filename is not None:
            with open(filename, "w") as fh:
                for msg in msgs:
                    fh.write(f"{msg}\n")
        return

    def summary(self, filename=None):
        msgs = []
        if self.SAVED_DTTMS == []:
            msgs.append("Nothing to show!")
            self.output(msgs, filename=filename)
            return

        msgs.append(f"{self.SAVED_DTTMS[0][0]} : {self.SAVED_DTTMS[0][1]}")
        for ((d0, t0), (d1, t1)) in zip(self.SAVED_DTTMS, self.SAVED_DTTMS[1:]):
            diff = (d1 - d0).total_seconds()
            msgs.append(f"{d1} : {t1} : {diff} sec diff")

        tot_diff = (self.SAVED_DTTMS[-1][0] - self.SAVED_DTTMS[0][0]).total_seconds()
        msgs.append(f"Total elapsed: {tot_diff}")

        self.output(msgs, filename=filename)

        return