class Person:
    def __init__(self, centroid, position):
        self.centroid = centroid
        self.previousPoint = centroid
        self.predictionPoint = centroid
        self.position = position
        self.crossed = False

    def get_array(self):
        return [self.centroid, self.previousPoint, self.predictionPoint, self.position, self.crossed]
