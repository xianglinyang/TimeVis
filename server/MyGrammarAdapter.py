from MyGrammar.MyGrammarListener import MyGrammarListener


class MyGrammarPrintListener(MyGrammarListener):
    def __init__(self, epochs=[1], selected_epoch=None):
        self.epochs = epochs
        if selected_epoch:
            self.selected_epoch = selected_epoch
        else:
            self.selected_epoch = self.epochs[0]
        self.result = ""
        self.pred_sample_needed = False
        self.epoch_sample_needed = False
        self.deltab_sample_needed = False
        self.stack = []
        self.array = []

    def enterMultiplecond2(self, ctx):
        if ctx.CONDOP().getText() == "&":
            self.result += " AND "
        elif ctx.CONDOP().getText() == "|":
            self.result += " OR "

    def enterParencond1(self, ctx):
        self.result += "("

    def exitParencond1(self, ctx):
        self.result += ")"

    def exitCond2(self, ctx):
        result = ""
        right = self.stack.pop()
        if self.stack:
            left  = self.stack.pop()
            result =  left + str(ctx.OP()) + right
        elif self.array:
            left = right
            lenArray = len(self.array)
            for _ in range(lenArray):
                i = self.array.pop(0)
                result += i +","
            result = left + " IN (" + result[:-1] + ")"       
        self.result += result

    def exitArray(self, ctx):
        for i in ctx.INT():
            self.array.append(i.getText())

    def exitParameter(self, ctx):
        if ctx.STRING():
            self.stack.append(self.checkString(ctx.STRING().getText()))

    def exitPositive(self, ctx):
        if ctx.INT():
            self.stack.append(ctx.INT().getText())

    def exitNegative(self, ctx):
    	# Return the number of indexes for an array based on the negative integer value
        if ctx.INT():
            value = int("-"+ctx.INT().getText())
            for i in self.epochs[value:]:
                self.array.append(str(i))

    def exitExpr(self, ctx):
    	# MYSQL statement is built here
        if "search for samples" in ctx.ACTION().getText():
            action = "SELECT Sample.idx FROM Sample "
        if self.result:
            self.result = "WHERE " + self.result
        if self.pred_sample_needed or self.epoch_sample_needed or self.deltab_sample_needed:
            self.result = action + "INNER JOIN PredSample ON Sample.idx =  PredSample.idx " + self.result
        else:
            self.result = action + self.result
        if (self.pred_sample_needed or self.deltab_sample_needed) and not self.epoch_sample_needed:
            self.result += " AND PredSample.epoch=" + str(self.selected_epoch)
        elif self.epoch_sample_needed:
            self.result += " GROUP BY Sample.idx"
        self.result += ";"

    def checkString(self, string):
    	# check the strings and categorize them for MYSQL statement
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        if "pred" in string:
            result = string.split(".")
            self.pred_sample_needed = True
            return "PredSample."+result[1]
        elif "epoch" in string:
            result = string.split(".")
            self.epoch_sample_needed = True
            return "PredSample."+result[1]
        elif "deltab" in string:
            result = string.split(".")
            self.deltab_sample_needed = True
            return "PredSample."+result[1]
        elif "sample" in string:
            result = string.split(".")
            return "Sample."+result[1]
        elif string in classes:
            return str(classes.index(string))
        elif string in ["test","train"]:
            return "'"+string+"'"
        elif string == "false":
            return "0"
        elif string == "true":
            return "1"
        else:
            return string + " "