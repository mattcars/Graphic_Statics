import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

'''
INTERSECTION OF 2 LINES

m = array of slopes [m1, m2]
x0 = x coordinate of points [x1, x2]
y0 = y coordinates of points [y1, y2]
'''
def get_intersect_points(m, x0, y0):

    b=[]
    for i in range(len(m)):
        b.append(m[i]*-x0[i] + y0[i])

    B = np.array(b).reshape(2, 1)
    A = np.array([[-m[0], 1], [-m[1], 1]])
    X = la.solve(A, B)

    return (X[0][0], X[1][0])

'''
INTERSECTION OF A CIRCLE AND A LINE

m0 = slope of line
(x0, y0) = point on line
d = radius of circle
(xi, yi) = center of circle
'''
def get_intersect_cirlce(m0, x0, y0, d, xi, yi):
    b = m0*x0 - y0 + yi

    A = (1+m0**2)
    B = (-2*xi-2*m0*b)
    C = (xi**2 + b**2 -d**2)

    if B**2 - 4*A*C < 0:
        return (np.nan, np.nan) , (np.nan, np.nan)

    x1 = (-B + (B**2 - 4*A*C)**0.5 )/(2*A)
    y1 = m0*(x1-x0) + y0

    x2 = (-B - (B**2 - 4*A*C)**0.5 )/(2*A)
    y2 = m0*(x2-x0) + y0

    return (x1, y1), (x2, y2)


class funicular_polygon:
    load_line = np.array([])
    origin = (0, 0)
    X = []
    Y = []

    '''Set origin for the funicular form'''
    def set_origin(self, ox, oy):
        self.origin = (ox, oy)

    '''Set loads and build the load line'''
    def set_loads(self, loads):
        if type(loads) != np.ndarray:
            if type(loads[0]) != list:
                loads = np.array([0] + loads)
            else:
                loads=np.array([[0, 0]] + loads)
        else:
            loads = np.concatenate((np.zeros((1, 2)), loads), axis=0)


        if len(loads.shape) == 1:

            Z = np.zeros((len(loads), 2))
            Z[:, 1] = loads
            loads = Z

        self.load_line = np.cumsum(loads, axis=0)


    '''Return the slopes of each member'''
    def slopes(self):
        return (self.load_line[:, 1] - self.origin[1])/(self.load_line[:, 0] - self.origin[0])

    '''Return Forces of each member'''
    def forces(self):
        return ((self.load_line[:, 0]-self.origin[0])**2 + (self.load_line[:, 1]-self.origin[1])**2)**0.5

    '''
    Build Geometry of the Structure
    X = x-spacing of forces (if no X is input, the function will return the current X, Y values)
    start = start y height of point
    start_i = start point to construct form from
    '''
    def get_geometry(self, X=0, start=0, start_i=0):
        if type(X)==int:
            return self.X, self.Y
        if type(X) !=np.array:
            X = np.array(X)

        self.X = X
        self.Y = np.zeros(X.shape)

        m = self.slopes()

        self.Y[start_i] = start

        # forward pass
        for k in range(0, len(m)-start_i):
            i = start_i + k + 1
            self.Y[i] = m[i-1] * (self.X[i] - self.X[i-1]) + self.Y[i-1]

        # backward pass
        for k in range(start_i):
            i = start_i - k -1
            self.Y[i] = m[i] * (self.X[i] - self.X[i+1]) + self.Y[i+1]

        return self.Y

    '''
    Plot the funicular force polygons
    ax = axis to plot on
    '''
    def plot_force(self, ax=0, m_color='g'):
        if ax==0:
            fig, ax = plt.subplots()

        ax.plot(self.load_line[:, 0], self.load_line[:, 1], c='b')
        ax.scatter(self.load_line[:, 0], self.load_line[:, 1], c='r')
        ax.scatter(self.origin[0], self.origin[1], c='k')

        for l in self.load_line:
            ax.plot([self.origin[0], l[0]], [self.origin[1], l[1]], c=m_color)

        if ax==0: return fig

    '''
    Plot the form of the structure
    ax = axis to plot on, otherwise create new plot
    amin = minimum alpha of plot
    '''
    def plot_form(self, ax=0, amin=.1, m_color='b'):

        # Scale forces to alpha values
        N = self.forces()
        N = N/np.max(N) * (1-amin) + amin

        if ax==0:
            fig, ax = plt.subplots()

        ax.scatter(self.X, self.Y, c='r')

        for i in range(len(self.X)-1):
            ax.plot([self.X[i], self.X[i+1]], [self.Y[i], self.Y[i+1]], c=m_color, alpha = N[i])


        if ax==0: return fig


    '''
    GET Y VALUE ON A FORM AT A GIVEN X

    poly = funicular polygon object
    x = x location
    '''
    def get_Y(self, x):
        i = np.where(self.X<=x)[0][-1]
        mi = (self.Y[i+1]-self.Y[i])/(self.X[i+1]-self.X[i])
        y = mi * (x-self.X[i]) + self.Y[i]
        return y


    '''
    FIND INTERSECTION OF A LINE WITH THE LOAD LINE

    poly = funicular polygon object
    p = point in force space
    m = slope of line
    '''
    def load_line_intersect(self, p, m):
        Y= m*(self.load_line[:, 0] - p[0]) + p[1]

        i1 = np.where(Y>=self.load_line[:, 1])[0][0]
        i0 = i1-1

        p0 = self.load_line[i0]
        p1 = self.load_line[i1]

        if p0[0] == p1[0]: inter = (p0[0], Y[i0])
        else:
            mLL = (p1[1]-p0[1])/(p1[0]-p0[0])
            xL = self.load_line[i0, 0]
            yL = self.load_line[i0, 1]
            inter = get_intersect_points([m, mLL], [p[0], xL], [p[1], yL])

        return inter


    '''
    FIND FORM OF VAULT GIVEN 3 POINTS
    '''
    def find_form(self, X_loc, X, Y, Z, max_iters=100, error=.01):

        # Slopes to build vault
        XY = (Y[1]-X[1])/(Y[0]-X[0])
        ZY = (Y[1]-Z[1])/(Y[0]-Z[0])

        self.origin = (np.min(self.load_line[:, 0])-2, self.load_line[-1, 1]/2)
        self.get_geometry(X_loc, start=X[1])

        zdiff = 1
        iters = 0
        while zdiff >= error and iters<max_iters:

            # Plot where Y falls onto this form
            y = (Y[0], self.get_Y(Y[0]))

            # Get this origin's slopes to y
            x = (self.X[0], self.Y[0])
            z = (self.X[-1], self.Y[-1])

            xy = (y[1]-x[1])/(y[0]-x[0])
            zy = (y[1]-z[1])/(y[0]-z[0])

            # Get intersections with load line
            m = self.load_line_intersect(self.origin, xy)
            n = self.load_line_intersect(self.origin, zy)

            # Use these points and the desired slopes to find new origin
            O = get_intersect_points([XY, ZY], [m[0], n[0]], [m[1], n[1]])

            # Get new funicular polygon
            self.origin = O
            self.get_geometry(X_loc, start=X[1])

            zdiff = abs(Z[1]-self.Y[-1])
            iters+=1


        return O

    '''
    FORM FIND GIVEN A MAXIMUM FORCE CONSTRAINT

    **ONLY WORKS ON VERTICAL LOADS, NO INCLINED LOADS**
    '''
    def get_origin(self, slope, Fmax):
        # error if there are inclined loads in form
        if np.sum(self.load_line[:, 0])!=0:
            print('Forms are not funicular because of inclined loads')
            print('\tmethod only works with vertical loads only')

        # center of force
        cF = [0, self.load_line[-1, 1]/2]

        origins = []
        for p in self.load_line:
            intersects = get_intersect_circle(slope, cF[0], cF[1], Fmax, p[0], p[1])

            for o in intersects:
                if np.nan not in o:
                    self.origin = o
                    if len(np.where(self.forces()-Fmax>10**-4)[0]) == 0 and o not in origins:
                        origins.append(o)

        return origins
