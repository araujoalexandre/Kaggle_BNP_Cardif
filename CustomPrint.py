"""
__file__

    CustomPrint.py

__description__

    Custom Print class for printing models scores
    
__author__

    Araujo Alexandre < alexandre.araujo@wavestone.com >

"""

class CustomPrint:

    def __init__(self, padding=14):
        self.headers = ['FOLD', 'TRAIN', 'CV', 'START', 'END', 'DUR']
        self.len_title = len(self.headers)
        self.padding = padding

    def to_print(self, string):
        print(string)

    def _format_timedelta(self, timedeltaObj):
        s = timedeltaObj.total_seconds()
        arr = [int(s // 3600), int(s % 3600 // 60), int(s % 60)]
        return '{:02d}:{:02d}:{:02d}'.format(*arr)

    def _title(self):
        arr = ["|{x: ^{fill}}".format(x=x, fill=self.padding).format(x) 
                                                          for x in self.headers]
        string = ''.join(arr) + "|"
        self.to_print('{}\n{}\n{}'.format(self._line(), string, self._line()))

    def score(self, fold, train, cv, start, end):
        if fold == 0:
            self._title()
        if fold == '':
            self.to_print(self._line())
        dur = self._format_timedelta(end - start)
        arr = [ fold, '{:.5f}'.format(train), '{:.5f}'.format(cv), 
                start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S"), dur]
        padding = self.padding
        arr = ["|{x: ^{fill}}".format(x=x, fill=padding).format(x) for x in arr]
        string = ''.join(arr) + "|"
        self.to_print(string)
        if fold == '':
            self.to_print(self._line())

    def _line(self):
        return "-" * (self.padding * self.len_title + self.len_title + 1)