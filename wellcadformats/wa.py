import csv


class Comments(object):
    def __init__(self, intervals=None):
        self.intervals = intervals

    def to_wac(self, file, title="Comments"):
        if isinstance(file, str):
            file = open(file, "w")
        table = zip(*self.intervals)
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["TopDepth", "BottomDepth", title])
        writer.writerow(["m", "m", ""])
        for row in self.intervals:
            writer.writerow(row)
        file.close()
