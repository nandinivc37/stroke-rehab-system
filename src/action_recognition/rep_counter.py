class RepCounter:
    def __init__(self):
        self.reps = 0
        self.stage = "up"

    def update(self, angle):
        """
        Updates repetition count using elbow angle.

        Returns:
            reps (int)
            stage (str): 'up' or 'down'
        """

        if angle < 50:
            self.stage = "down"

        elif angle > 160 and self.stage == "down":
            self.stage = "up"
            self.reps += 1

        return self.reps, self.stage