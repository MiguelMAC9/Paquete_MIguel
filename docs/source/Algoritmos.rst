==========
Algoritmos
==========

Este documento describe varios algoritmos de optimización implementados en Python. A continuación se presentan los algoritmos, junto con una explicación de su funcionamiento y los requisitos para su uso.

Random Walk
===========

**Función**: `random_walk_2d`

.. code-block:: python

    import random

    def random_walk_2d(n_steps, initial_position, mu=0, sigma=1):
        x, y = [initial_position[0]], [initial_position[1]]  # Posición inicial
        
        for _ in range(n_steps):
            step_x = random.gauss(mu, sigma)  # Incremento aleatorio en x
            step_y = random.gauss(mu, sigma)  # Incremento aleatorio en y
            x.append(x[-1] + step_x)
            y.append(y[-1] + step_y)
        return x, y

**Descripción**: Este algoritmo realiza un paseo aleatorio en dos dimensiones, comenzando desde una posición inicial y dando un número especificado de pasos aleatorios en la dirección x e y. Los incrementos en cada dirección son valores aleatorios generados a partir de una distribución gaussiana.

**Requisitos**:
   - `n_steps`: Número de pasos a realizar.
   - `initial_position`: Posición inicial (x, y).
   - `mu`: Media de la distribución gaussiana (por defecto 0).
   - `sigma`: Desviación estándar de la distribución gaussiana (por defecto 1).

Nelder-Mead Simplex
===================

**Función**: `NelderMead_Simplex`

.. code-block:: python

    def NelderMead_Simplex(funcion, x0, tol=1e-6, iteraciones=100, alpha=1, beta=0.5, gamma=2):
        n = len(x0)
        vectores = [x0[:]]  # Crear una lista de listas
        for i in range(n):
            nuevo_vector = x0[:]
            nuevo_vector[i] += 0.05
            vectores.append(nuevo_vector)
        
        valores_vect = [funcion(v) for v in vectores]

        for _ in range(iteraciones):
            ordenar_vect = sorted(range(len(valores_vect)), key=lambda k: valores_vect[k])
            vectores = [vectores[i] for i in ordenar_vect]
            valores_vect = [valores_vect[i] for i in ordenar_vect]

            xc = [sum(vectores[j][i] for j in range(n)) / n for i in range(n)]

            xr = [xc[i] + alpha * (xc[i] - vectores[-1][i]) for i in range(n)]
            fr = funcion(xr)
            if valores_vect[-2] > fr >= valores_vect[0]:
                vectores[-1], valores_vect[-1] = xr, fr
            elif fr < valores_vect[0]:
                expansion = [xc[i] + gamma * (xr[i] - xc[i]) for i in range(n)]
                fe = funcion(expansion)
                if fe < fr:
                    vectores[-1], valores_vect[-1] = expansion, fe
                else:
                    vectores[-1], valores_vect[-1] = xr, fr
            else:
                contraccion = [xc[i] + beta * (vectores[-1][i] - xc[i]) for i in range(n)]
                fc = funcion(contraccion)
                if fc < valores_vect[-1]:
                    vectores[-1], valores_vect[-1] = contraccion, fc
                else:
                    for i in range(1, len(vectores)):
                        for j in range(n):
                            vectores[i][j] = (vectores[0][j] + vectores[i][j]) / 2
                        valores_vect[i] = funcion(vectores[i])

        return vectores[0]

**Descripción**: El método Nelder-Mead Simplex es un algoritmo de optimización sin derivadas que utiliza un simplex (un polígono de n+1 vértices en n dimensiones) para buscar el mínimo de una función. El simplex se ajusta mediante operaciones de reflexión, expansión y contracción para moverse hacia el mínimo de la función.

**Requisitos**:
   - `funcion`: Función objetivo a minimizar.
   - `x0`: Vector inicial.
   - `tol`: Tolerancia para la convergencia.
   - `iteraciones`: Número máximo de iteraciones.
   - `alpha`, `beta`, `gamma`: Parámetros del algoritmo para reflexión, contracción y expansión.

Hooke-Jeeves
============

**Función**: `hooke_jeeves`

.. code-block:: python

    def Busqueda(x, d, funcion, limite=1e10):
        x_i = x[:]
        for i in range(len(x)):
            for direction in [-1, 1]:
                x_t = x_i[:]
                x_t[i] += direction * d
                if abs(x_t[i]) > limite:
                    continue
                if funcion(x_t) < funcion(x_i):
                    x_i = x_t
        return x_i

    def hooke_jeeves(x_i, delta, alpha, e, n_iter, funcion, limite=1e10):
        x_b = x_i[:]
        x_m = x_b[:]
        iter_c = 0
        resul = [x_b[:]]

        while delta > e and iter_c < n_iter:
            x_n = Busqueda(x_b, delta, funcion, limite)
            if funcion(x_n) < funcion(x_m):
                x_b = [2 * x_n[i] - x_m[i] for i in range(len(x_n))]
                x_m = x_n[:]
            else:
                delta *= alpha
                x_b = x_m[:]
            resul.append(x_b[:])
            iter_c += 1

        return x_m, resul

**Descripción**: El método Hooke-Jeeves es un algoritmo de optimización sin derivadas que realiza una búsqueda de patrones. Se mueve en la dirección de la mejora y ajusta el tamaño del paso hasta que se encuentra un mínimo local.

**Requisitos**:
   - `x_i`: Vector inicial.
   - `delta`: Tamaño inicial del paso.
   - `alpha`: Factor de reducción del tamaño del paso.
   - `e`: Tolerancia para la convergencia.
   - `n_iter`: Número máximo de iteraciones.
   - `funcion`: Función objetivo a minimizar.
   - `limite`: Límite en las variables.

Cauchy
======

**Función**: `cauchy`

.. code-block:: python

    import math

    def gradiente(f, x, deltaX=0.001):
        grad = []
        for i in range(len(x)):
            xp = x[:]
            xn = x[:]
            xp[i] += deltaX
            xn[i] -= deltaX
            grad.append((f(xp) - f(xn)) / (2 * deltaX))
        return grad

    def cauchy(funcion, x0, epsilon1, epsilon2, M, optimizador_univariable):
        terminar = False
        xk = x0[:]
        k = 0
        while not terminar:
            grad = gradiente(funcion, xk)

            if math.sqrt(sum(g**2 for g in grad)) < epsilon1 or k >= M:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return funcion([xk[i] - alpha * grad[i] for i in range(len(xk))])

                alpha = optimizador_univariable(alpha_funcion, epsilon2, a=0.0, b=1.0)
                x_k1 = [xk[i] - alpha * grad[i] for i in range(len(xk))]
                print(xk, alpha, grad, x_k1)

                if math.sqrt(sum((x_k1[i] - xk[i])**2 for i in range(len(xk)))) / (math.sqrt(sum(xi**2 for xi in xk)) + 0.00001) <= epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1
        return xk

**Descripción**: El método de Cauchy es un algoritmo de optimización que utiliza el gradiente de la función objetivo para buscar el mínimo. La dirección de descenso se determina a partir del gradiente y se ajusta el tamaño del paso utilizando un optimizador univariable.

**Requisitos**:
   - `funcion`: Función objetivo a minimizar.
   - `x0`: Vector inicial.
   - `epsilon1`, `epsilon2`: Tolerancias para la convergencia.
   - `M`: Número máximo de iteraciones.
   - `optimizador_univariable`: Método de optimización para determinar el tamaño del paso.

Método de Fletcher-Reeves
=========================

**Función**: `fletcher_reeves`

.. code-block:: python

    def gradient(f, x, deltaX=1e-5):
        grad = [0] * len(x)
        for i in range(len(x)):
            x1 = x[:]
            x2 = x[:]
            x1[i] += deltaX
            x2[i] -= deltaX
            grad[i] = (f(x1) - f(x2)) / (2 * deltaX)
        return grad

    def fletcher_reeves(funcion, x0, epsilon1, epsilon2, M, optimizador_univariable):
        terminar = False
        xk = x0[:]
        gk = gradient(funcion, xk)
        dk = [-g for g in gk]
        k = 0

        while not terminar:
            def phi(alpha):
                return funcion([xk[i] + alpha * dk[i] for i in range(len(xk))])

            alpha = optimizador_univariable(phi, epsilon2, a=0.0, b=1.0)
            x_k1 = [xk[i] + alpha * dk[i] for i in range(len(xk))]

            g_k1 = gradient(funcion, x_k1)
            beta = sum(g_k1[i]**2 for i in range(len(g_k1))) / sum(gk[i]**2 for i in range(len(gk)))
            d_k1 = [-g_k1[i] + beta * dk[i] for i in range(len(gk))]

            if math.sqrt(sum(g**2 for g in g_k1)) < epsilon1 or k >= M:
                terminar = True
            else:
                if math.sqrt(sum((x_k1[i] - xk[i])**2 for i in range(len(xk)))) / (math.sqrt(sum(xi**2 for xi in xk)) + 1e-5) < epsilon2:
                    terminar = True
                else:
                    xk, gk, dk = x_k1, g_k1, d_k1
                    k += 1
        return xk

**Descripción**: El método de Fletcher-Reeves es un algoritmo de optimización basado en gradientes, que pertenece a la familia de métodos de conjugado gradiente. Utiliza la dirección del gradiente y una actualización conjugada para encontrar el mínimo de la función objetivo.

**Requisitos**:
   - `funcion`: Función objetivo a minimizar.
   - `x0`: Vector inicial.
   - `epsilon1`, `epsilon2`: Tolerancias para la convergencia.
   - `M`: Número máximo de iteraciones.
   - `optimizador_univariable`: Método de optimización para determinar el tamaño del paso.

Método de Newton
================

**Función**: `Newton`

.. code-block:: python

    import numpy as np

    def hessiana(f, x, h=1e-5):
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_ij1, x_ij2, x_ij3, x_ij4 = x[:], x[:], x[:], x[:]
                x_ij1[i] += h
                x_ij1[j] += h
                x_ij2[i] += h
                x_ij2[j] -= h
                x_ij3[i] -= h
                x_ij3[j] += h
                x_ij4[i] -= h
                x_ij4[j] -= h
                hess[i, j] = (f(x_ij1) - f(x_ij2) - f(x_ij3) + f(x_ij4)) / (4 * h**2)
        return hess

    def gradiente(f, x, h=1e-5):
        grad = np.zeros(len(x))
        for i in range(len(x)):
            x1, x2 = x[:], x[:]
            x1[i] += h
            x2[i] -= h
            grad[i] = (f(x1) - f(x2)) / (2 * h)
        return grad

    def Newton(funcion, x0, epsilon1, epsilon2, M):
        xk = np.array(x0)
        for k in range(M):
            grad = gradiente(funcion, xk)
            hess = hessiana(funcion, xk)

            if np.linalg.norm(grad) < epsilon1:
                break

            dk = np.linalg.solve(hess, -grad)
            alpha = 1.0
            x_k1 = xk + alpha * dk

            if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 1e-5) < epsilon2:
                break

            xk = x_k1

        return xk

**Descripción**: El método de Newton es un algoritmo de optimización basado en derivadas, que utiliza la información del gradiente y la matriz hessiana de la función objetivo para encontrar el mínimo. Este método es conocido por su rápida convergencia cerca de un mínimo.

**Requisitos**:
   - `funcion`: Función objetivo a minimizar.
   - `x0`: Vector inicial.
   - `epsilon1`, `epsilon2`: Tolerancias para la convergencia.
   - `M`: Número máximo de iteraciones.


Método de División de Intervalos por la Mitad
=============================================

**Función**: `bisection_method`

.. code-block:: python

    import math

    def bisection_method(f, a, b, tol=1e-6, max_iter=1000):
        if f(a) * f(b) >= 0:
            raise ValueError("La función no cambia de signo en el intervalo dado [a, b].")

        left = a
        right = b

        for i in range(max_iter):
            midpoint = (left + right) / 2.0
            f_mid = f(midpoint)

            if abs(f_mid) < tol:
                print(f'Convergencia alcanzada en {i+1} iteraciones')
                return midpoint

            if f(left) * f_mid < 0:
                right = midpoint
            else:
                left = midpoint

        raise ValueError(f'El método de bisección no converge después de {max_iter} iteraciones.')

**Descripción**: El método de bisección es un algoritmo de búsqueda de raíces que divide repetidamente un intervalo por la mitad y selecciona el subintervalo que contiene la raíz. Se repite este proceso hasta que se alcanza una tolerancia especificada o el número máximo de iteraciones.

**Requisitos**:
- `f`: Función objetivo para la cual se busca la raíz.
- `a`, `b`: Extremos del intervalo inicial.
- `tol`: Tolerancia para la convergencia.
- `max_iter`: Número máximo de iteraciones.

Búsqueda de Fibonacci
=====================

**Función**: `fibonacci_search`

.. code-block:: python

    def fibonacci_search(f, a, b, n, tol=1e-6):
        fib = [0, 1]
        for i in range(2, n+1):
            fib.append(fib[i-1] + fib[i-2])

        L = b - a

        for k in range(1, n):
            x1 = a + (fib[n-k-1] / fib[n-k+1]) * L
            x2 = a + (fib[n-k] / fib[n-k+1]) * L
            fx1 = f(x1)
            fx2 = f(x2)

            if fx1 < fx2:
                b = x2
            else:
                a = x1

            L = b - a

            if L < tol:
                return (a + b) / 2

        return (a + b) / 2

**Descripción**: La búsqueda de Fibonacci es un método de optimización que utiliza números de Fibonacci para reducir el intervalo de búsqueda y encontrar el mínimo de una función unimodal en un intervalo cerrado.

**Requisitos**:
- `f`: Función objetivo a minimizar.
- `a`, `b`: Extremos del intervalo inicial.
- `n`: Número de iteraciones basadas en la secuencia de Fibonacci.
- `tol`: Tolerancia para la convergencia.

Método de la Sección Dorada
===========================

**Función**: `golden_section_search`

.. code-block:: python

    def golden_section_search(f, a, b, tol=1e-6):
        phi = (math.sqrt(5) - 1) / 2
        x1 = a + (1 - phi) * (b - a)
        x2 = a + phi * (b - a)
        fx1 = f(x1)
        fx2 = f(x2)

        while abs(b - a) > tol:
            if fx1 < fx2:
                b = x2
                x2 = x1
                fx2 = fx1
                x1 = a + (1 - phi) * (b - a)
                fx1 = f(x1)
            else:
                a = x1
                x1 = x2
                fx1 = fx2
                x2 = a + phi * (b - a)
                fx2 = f(x2)

        return (a + b) / 2

**Descripción**: El método de la sección dorada es un algoritmo de búsqueda que utiliza la proporción áurea para encontrar el mínimo de una función unimodal dentro de un intervalo cerrado, reduciendo sistemáticamente el intervalo de búsqueda.

**Requisitos**:
- `f`: Función objetivo a minimizar.
- `a`, `b`: Extremos del intervalo inicial.
- `tol`: Tolerancia para la convergencia.

Método de Newton-Raphson
========================

**Función**: `newton_raphson`

.. code-block:: python

    def newton_raphson(f, df, x0, tol=1e-6, max_iter=1000):
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < tol:
                print(f'Convergencia alcanzada en {i+1} iteraciones')
                return x
            dfx = df(x)
            if dfx == 0:
                raise ValueError("Derivada de la función es cero. No se puede continuar.")
            x = x - fx / dfx
        raise ValueError(f'El método de Newton-Raphson no converge después de {max_iter} iteraciones.')

**Descripción**: El método de Newton-Raphson es un algoritmo de búsqueda de raíces que utiliza derivadas para aproximar la raíz de una función. Se basa en iterar una fórmula que converge cuadráticamente hacia la raíz.

**Requisitos**:
- `f`: Función objetivo para la cual se busca la raíz.
- `df`: Derivada de la función objetivo.
- `x0`: Valor inicial.
- `tol`: Tolerancia para la convergencia.
- `max_iter`: Número máximo de iteraciones.

Método de la Secante
====================

**Función**: `secant_method`

.. code-block:: python

    def secant_method(f, x0, x1, tol=1e-6, max_iter=1000):
        fx0 = f(x0)
        fx1 = f(x1)

        for i in range(max_iter):
            if abs(fx1) < tol:
                print(f'Convergencia alcanzada en {i+1} iteraciones')
                return x1

            x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

            x0 = x1
            x1 = x_next
            fx0 = fx1
            fx1 = f(x_next)

        raise ValueError(f'El método de la secante no converge después de {max_iter} iteraciones.')

**Descripción**: El método de la secante es un algoritmo de búsqueda de raíces que utiliza dos aproximaciones iniciales y no requiere el cálculo de derivadas. Es una versión modificada del método de Newton-Raphson.

**Requisitos**:
- `f`: Función objetivo para la cual se busca la raíz.
- `x0`, `x1`: Aproximaciones iniciales.
- `tol`: Tolerancia para la convergencia.
- `max_iter`: Número máximo de iteraciones.

Método de Bisección
===================

**Función**: `bisection`

.. code-block:: python

    def bisection(f, a, b, tol=1e-6, max_iter=1000):
        if f(a) * f(b) >= 0:
            raise ValueError("La función no cambia de signo en el intervalo dado.")

        for i in range(max_iter):
            c = (a + b) / 2.0
            fc = f(c)

            if abs(fc) < tol:
                print(f'Convergencia alcanzada en {i+1} iteraciones')
                return c

            if f(a) * fc < 0:
                b = c
            else:
                a = c

        raise ValueError(f'El método de bisección no converge después de {max_iter} iteraciones.')

**Descripción**: El método de bisección es un algoritmo de búsqueda de raíces que divide repetidamente un intervalo por la mitad y selecciona el subintervalo que contiene la raíz. Se repite este proceso hasta que se alcanza una tolerancia especificada o el número máximo de iteraciones.

**Requisitos**:
- `f`: Función objetivo para la cual se busca la raíz.
- `a`, `b`: Extremos del intervalo inicial.
- `tol`: Tolerancia para la convergencia.
- `max_iter`: Número máximo de iteraciones.
