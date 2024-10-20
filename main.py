from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from sklearn import svm



class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        # Create a canvas widget
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        # Draw grid
        self.draw_grid()

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Button to get matrix
        self.get_matrix_button = tk.Button(root, text="Get Matrix", command=self.get_matrix)
        self.get_matrix_button.pack()

        # Track filled pixels
        self.filled_pixels = np.zeros((28, 28), dtype=np.uint8)

    def draw_grid(self):
        """ Draws a 28x28 grid on the canvas. """
        for i in range(29):
            self.canvas.create_line(i * 10, 0, i * 10, 280, fill="lightgray")
            self.canvas.create_line(0, i * 10, 280, i * 10, fill="lightgray")

    def paint(self, event):
        """ Paints on the canvas and marks the corresponding pixel as filled. """
        x, y = event.x, event.y
        col = x // 10
        row = y // 10
        self.canvas.create_rectangle(col * 10, row * 10, (col + 1) * 10, (row + 1) * 10, fill="black", outline="black")
        self.filled_pixels[row, col] = 1

    def get_matrix(self):
        """ Converts the filled pixels to a 28x28 matrix with values 0 or 255. """
        matrix = self.filled_pixels * 255
        return matrix
        # You can also save or process the matrix here


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()



mnist = fetch_openml("mnist_784")
x,y = mnist['data'],mnist['target']

h = app.get_matrix()
some_digit = np.array(h)
print(some_digit)
some_digit_image=some_digit.reshape(28, 28)
some_digit_flattened = some_digit.ravel()

x_train= np.array(x.iloc[:])
y_train = np.array(y.iloc[:])

shuffle_index=np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train=y_train.astype(np.int8)

clf = svm.SVC()
clf.fit(x_train, y_train)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.show()

ans = clf.predict([some_digit_flattened])
print(ans)
