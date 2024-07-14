Funciones
=========

Este documento describe varias funciones matemáticas utilizadas en optimización y otros campos. A continuación se presentan las implementaciones y descripciones detalladas de cada función.

Funciones
---------

.. code-block:: python

    import math

    def himmelblau(x):
        """
        Función de Himmelblau.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Himmelblau evaluado en x.
        """
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def testfunction(x):
        """
        Función de prueba simple.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: La suma de los cuadrados de los elementos de x.
        """
        return x[0]**2 + x[1]**2

    def sphere(x):
        """
        Función esfera.
        
        Parámetros:
        x (list): Lista de n elementos.

        Retorna:
        float: La suma de los cuadrados de los elementos de x.
        """
        return sum(xi**2 for xi in x)

    def rastrigin(x, A=10):
        """
        Función de Rastrigin.
        
        Parámetros:
        x (list): Lista de n elementos.
        A (float): Constante, por defecto 10.

        Retorna:
        float: El valor de la función Rastrigin evaluado en x.
        """
        return A * len(x) + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)

    def rosenbrock(x):
        """
        Función de Rosenbrock.
        
        Parámetros:
        x (list): Lista de n elementos.

        Retorna:
        float: El valor de la función Rosenbrock evaluado en x.
        """
        return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))

    def beale(x):
        """
        Función de Beale.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Beale evaluado en x.
        """
        return ((1.5 - x[0] + x[0] * x[1])**2 +
                (2.25 - x[0] + x[0] * x[1]**2)**2 +
                (2.625 - x[0] + x[0] * x[1]**3)**2)

    def goldstein(x):
        """
        Función de Goldstein-Price.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Goldstein-Price evaluado en x.
        """
        part1 = (1 + (x[0] + x[1] + 1)**2 * 
                (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2))
        part2 = (30 + (2 * x[0] - 3 * x[1])**2 * 
                (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
        return part1 * part2

    def boothfunction(x):
        """
        Función de Booth.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Booth evaluado en x.
        """
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

    def bunkinn6(x):
        """
        Función Bunkin N.6.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Bunkin N.6 evaluado en x.
        """
        return 100 * math.sqrt(abs(x[1] - 0.001 * x[0]**2)) + 0.01 * abs(x[0] + 10)

    def matyas(x):
        """
        Función de Matyas.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Matyas evaluado en x.
        """
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def levi(x):
        """
        Función de Levi.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Levi evaluado en x.
        """
        part1 = math.sin(3 * math.pi * x[0])**2
        part2 = (x[0] - 1)**2 * (1 + math.sin(3 * math.pi * x[1])**2)
        part3 = (x[1] - 1)**2 * (1 + math.sin(2 * math.pi * x[1])**2)
        return part1 + part2 + part3

    def threehumpcamel(x):
        """
        Función de camello de tres jorobas.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función de camello de tres jorobas evaluado en x.
        """
        return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

    def easom(x):
        """
        Función de Easom.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Easom evaluado en x.
        """
        return -math.cos(x[0]) * math.cos(x[1]) * math.exp(-(x[0] - math.pi)**2 - (x[1] - math.pi)**2)

    def crossintray(x):
        """
        Función Cross-in-Tray.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Cross-in-Tray evaluado en x.
        """
        op = abs(math.sin(x[0]) * math.sin(x[1]) * math.exp(abs(100 - math.sqrt(x[0]**2 + x[1]**2) / math.pi)))
        return -0.0001 * (op + 1)**0.1

    def eggholder(x):
        """
        Función de Eggholder.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Eggholder evaluado en x.
        """
        op1 = -(x[1] + 47) * math.sin(math.sqrt(abs(x[0] / 2 + (x[1] + 47))))
        op2 = -x[0] * math.sin(math.sqrt(abs(x[0] - (x[1] + 47))))
        return op1 + op2

    def holdertable(x):
        """
        Función de Holder Table.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Holder Table evaluado en x.
        """
        op = abs(math.sin(x[0]) * math.cos(x[1]) * math.exp(abs(1 - math.sqrt(x[0]**2 + x[1]**2) / math.pi)))
        return -op

    def mccormick(x):
        """
        Función de McCormick.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función McCormick evaluado en x.
        """
        return math.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

    def schaffern2(x):
        """
        Función de Schaffer N.2.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Schaffer N.2 evaluado en x.
        """
        numerator = math.sin(x[0]**2 - x[1]**2)**2 - 0.5
        denominator = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + numerator / denominator

    def schaffern4(x):
        """
        Función de Schaffer N.4.
        
        Parámetros:
        x (list): Lista de dos elementos [x1, x2].

        Retorna:
        float: El valor de la función Schaffer N.4 evaluado en x.
        """
        num = math.cos(math.sin(abs(x[0]**2 - x[1]**2))) - 0.5
        den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + num / den

    def styblinskitang(x):
        """
        Función de Styblinski-Tang.
        
        Parámetros:
        x (list): Lista de n elementos.

        Retorna:
        float: El valor de la función Styblinski-Tang evaluado en x.
        """
        return sum((xi**4 - 16 * xi**2 + 5 * xi) / 2 for xi in x)

    def shekel(x, a=None, c=None):
        """
        Función de Shekel.

        Parámetros:
        x (list): Lista de dos elementos [x1, x2].
        a (list, optional): Matriz de coeficientes. Si no se proporciona, se usa una matriz predeterminada.
        c (list, optional): Lista de constantes. Si no se proporciona, se usa una lista predeterminada.

        Retorna:
        float: El valor de la función Shekel evaluado en x.
        """
        if a is None:
            a = [
                [4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, 1.0, 1.0],
                [8.0, 8.0, 8.0, 8.0],
                [6.0, 6.0, 6.0, 6.0],
                [3.0, 7.0, 3.0, 7.0],
                [2.0, 9.0, 2.0, 9.0],
                [5.0, 5.0, 3.0, 3.0],
                [8.0, 1.0, 8.0, 1.0],
                [6.0, 2.0, 6.0, 2.0],
                [7.0, 3.6, 7.0, 3.6]
            ]
        if c is None:
            c = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
            
        m = len(c)
        s = 0
        for i in range(m):
            s -= 1 / (sum((x[j] - a[i][j])**2 for j in range(2)) + c[i])
        return s
