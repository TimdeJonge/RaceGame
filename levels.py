from Polygon import Polygon
from Circle import Circle
from Rectangle import Rectangle
X = 10000
baby_park = [Polygon([(400, 150), (800, 150), (1000, 300), (1000, 500),
                      (800, 650), (400, 650), (200, 500), (200, 300)]),
             Rectangle([-X, -X], [0, X]),
             Rectangle([1200, -X], [X, X]),
             Rectangle([-X, 0], [X, -X]),
             Rectangle([-X, X], [X, 800])]
circles = [Circle((300, 300), 50),
           Circle((500, 500), 50),
           Circle((300, 700), 50),
           Polygon([(0, -X), (-X, -X), (-X, X), (0, X)])]

l = [Polygon([(0, -X), (0, X), (-X, X), (-X, -X)]),
     Polygon([(-X, 0), (X, 0), (X, -X), (-X, -X)]),
     Polygon([(500, -X), (X, -X), (X, 450), (500, 450)]),
     Circle([550, 450], 50),
     Polygon([(550, -X), (X, -X), (X, 500), (550, 500)]),
     Polygon([(1300, 450), (X, 450), (X, X), (1300, X)]),
     Polygon([(-X, 1000), (X, 1000), (X, X), (-X, X)]),
     Circle([250, 250], 150),
     Polygon([(100, 250), (400, 250), (400, 800), (100, 800)]),
     Circle([200, 800], 100),
     Polygon([(200, 600), (1050, 600), (1050, 900), (200, 900)]),
     Circle((1050, 750), 150)]

clear = []

test = [Rectangle([0, 0], [100, 100])]
