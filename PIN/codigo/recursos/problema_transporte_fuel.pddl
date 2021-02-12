;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; PROBLEMA TRANSPORTE                                      ;;;
;;; Ejercicio 2: Dominio Temporal                            ;;;
;;;                                                          ;;;
;;; Planificacion Inteligente @ MIARFID, UPV                 ;;;
;;;                                                          ;;;
;;; Autores:                                                 ;;;
;;;         Jaime Ferrando Huertas                           ;;;
;;;         Javier Martínez Bernia                           ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (problem probtransporte)
    (:domain transporte)
    (:objects
        CIUDAD1 CIUDAD2 CIUDAD3 - ciudad
        Ca1 Ca2 Ca3 - casa
        E1 E2 E3 E4 - estacion
        A1 A2 - aeropuerto
        F1 F2 F3 F4 - furgoneta
        T1 T2 - tren
        Av1 Av2 - avion
        P1 P2 - paquete
        D1 D2 D3 - conductor
    )

    (:init
        (in CIUDAD1 Ca1)
        (in CIUDAD1 A1)
        (in CIUDAD1 E1)
        (in CIUDAD1 E2)
        (at F1 E1)
        (at F2 A1)
        (at Av1 A1)
        (at T1 E2)
        (at P1 Ca1)
        (at D1 E2)
        (empty F1)
        (empty F2)

        (in CIUDAD2 Ca2)
        (in CIUDAD2 A2)
        (in CIUDAD2 E3)
        (at F3 A2)
        (at Av2 A2)
        (at P2 Ca2)
        (at D2 Ca2)
        (empty F3)

        (in CIUDAD3 Ca3)
        (in CIUDAD3 E4)
        (at F4 Ca3)
        (at T2 E4)
        (at D3 Ca3)
        (empty F4)

        ; Temporal
        (= (peso P1) 10)
        (= (peso P2) 15)
        ; Ciudad 1
        (= (distancia Ca1 E1) 50)
        (= (distancia E1 Ca1) 50)
        (= (distancia Ca1 E2) 120)
        (= (distancia E2 Ca1) 120)
        (= (distancia Ca1 A1) 50)
        (= (distancia A1 Ca1) 50)
        (= (distancia A1 E1) 100)
        (= (distancia E1 A1) 100)
        (= (distancia A1 E2) 60)
        (= (distancia E2 A1) 60)
        (= (distancia E2 E1) 50)
        (= (distancia E1 E2) 50)

        ;; Distancias aeropuertos
        (= (distancia A1 A2) 60)
        (= (distancia A2 A1) 60)
        ;; Distancias estaciones
        (= (distancia E1 E2) 60)
        (= (distancia E2 E1) 60)
        (= (distancia E1 E3) 60)
        (= (distancia E3 E1) 60)
        (= (distancia E1 E4) 60)
        (= (distancia E4 E1) 60)

        (= (distancia E2 E3) 60)
        (= (distancia E3 E2) 60)
        (= (distancia E2 E4) 60)
        (= (distancia E4 E2) 60)

        (= (distancia E3 E4) 60)
        (= (distancia E4 E3) 60)

        ; Ciudad 2
        (= (distancia A2 E3) 150)
        (= (distancia E3 A2) 150)
        (= (distancia A2 Ca2) 80)
        (= (distancia Ca2 A2) 80)
        (= (distancia Ca2 E3) 150)
        (= (distancia E3 Ca2) 150)

        ; Ciudad 3
        (= (distancia E4 Ca3) 70)
        (= (distancia Ca3 E4) 70)

        ; Velocidades
        (= (velocidad F1) 50)
        (= (velocidad F2) 50)
        (= (velocidad F3) 50)
        (= (velocidad F4) 50)
        (= (velocidad D1) 10)
        (= (velocidad D2) 10)
        (= (velocidad D3) 10)

        (= (velocidad Av1) 300)
        (= (velocidad Av2) 300)
        (= (velocidad T1) 100)
        (= (velocidad T2) 100)

        ;Combustible
        (= (combustibletotal) 0)

        ;Aviones
        ;Aviones
        (= (combustible Av1) 100)
        (= (combustible Av2) 100)
        ;Trenes
        (= (combustible T1) 100)
        (= (combustible T2) 100)
        ;Furgonetas
        (= (combustible F1) 100)
        (= (combustible F2) 100)
        (= (combustible F3) 100)
        (= (combustible F4) 100)

        ;Capacidad
        (= (capacidad Av1) 1000)
        (= (capacidad Av2) 1000)
        (= (capacidad T1) 500)
        (= (capacidad T2) 500)
        (= (capacidad F1) 250)
        (= (capacidad F2) 250)
        (= (capacidad F3) 250)
        (= (capacidad F4) 250)

        ; ;Gasto
        (= (gasto Av1) 5)
        (= (gasto Av2) 5)
        (= (gasto T1) 10)
        (= (gasto T2) 10)
        (= (gasto F1) 14)
        (= (gasto F2) 14)
        (= (gasto F3) 14)
        (= (gasto F4) 14)

    )

    (:goal
        (and
            (at P1 Ca3) (at P2 Ca3)
        )
    )

    ; (:metric minimize (total-time))
    (:metric minimize
        (combustibletotal)
    )
)