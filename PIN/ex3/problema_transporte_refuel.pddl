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

(define (problem probtransporte) (:domain transporte)
(:objects CIUDAD1 CIUDAD2 CIUDAD3 - ciudad
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
    (in CIUDAD1 Ca1) (in CIUDAD1 A1) (in CIUDAD1 E1) (in CIUDAD1 E2)
    (at F1 E1) (at F2 A1) (at Av1 A1) (at T1 E2) (at P1 Ca1) (at D1 E2)
    (empty F1) (empty F2)

    (in CIUDAD2 Ca2) (in CIUDAD2 A2) (in CIUDAD2 E3)
    (at F3 A2) (at Av2 A2) (at P2 Ca2) (at D2 Ca2)
    (empty F3)

    (in CIUDAD3 Ca3) (in CIUDAD3 E4)
    (at F4 Ca3) (at T2 E4) (at D3 Ca3)
    (empty F4)

    ; Temporal
    (= (peso P1) 10) (= (peso P2) 15)
    ; Ciudad 1
    (= (distancia Ca1 E1) 50) (= (distancia E1 Ca1) 50)
    (= (distancia Ca1 E2) 120) (= (distancia E2 Ca1) 120)
    (= (distancia Ca1 A1) 50) (= (distancia A1 Ca1) 50)
    (= (distancia A1 E1) 100) (= (distancia E1 A1) 100)
    (= (distancia A1 E2) 60) (= (distancia E2 A1) 60)
    (= (distancia E2 E1) 50) (= (distancia E1 E2) 50)

    ;; Distancias aeropuertos
    (= (distancia A1 A2) 60) (= (distancia A2 A1) 60)
    ;; Distancias estaciones
    (= (distancia E1 E2) 60) (= (distancia E2 E1) 60)
    (= (distancia E1 E3) 60) (= (distancia E3 E1) 60)
    (= (distancia E1 E4) 60) (= (distancia E4 E1) 60)

    (= (distancia E2 E3) 60) (= (distancia E3 E2) 60)
    (= (distancia E2 E4) 60) (= (distancia E4 E2) 60)

    (= (distancia E3 E4) 60) (= (distancia E4 E3) 60)


    ; Ciudad 2
    (= (distancia A2 E3) 150) (= (distancia E3 A2) 150)
    (= (distancia A2 Ca2) 80) (= (distancia Ca2 A2) 80)
    (= (distancia Ca2 E3) 150) (= (distancia E3 Ca2) 150)

    ; Ciudad 3
    (= (distancia E4 Ca3) 70) (= (distancia Ca3 E4) 70)

    ; Velocidades
    (= (velocidad F1) 50) (= (velocidad F2) 50)
    (= (velocidad F3) 50) (= (velocidad F4) 50)
    (= (velocidad D1) 10) (= (velocidad D2) 10)
    (= (velocidad D3) 10)

    (= (velocidad Av1) 300)
    (= (velocidad Av2) 300)
    (= (velocidad T1) 100)
    (= (velocidad T2) 100)

    ; Combustible
    (= (combustibletotal) 0)

    ;Aviones
    (= (combustible Av1) 0)
    (= (combustible Av2) 0)
    ;Trenes
    (= (combustible T1) 0)
    (= (combustible T2) 0)
    ;Furgonetas
    (= (combustible F1) 0)
    (= (combustible F2) 0)
    (= (combustible F3) 0)
    (= (combustible F4) 0)

    ;Capacidad
    (= (capacidad Av1) 10000)
    (= (capacidad Av2) 10000)
    (= (capacidad T1) 5000)
    (= (capacidad T2) 5000)
    (= (capacidad F1) 1000)
    (= (capacidad F2) 1000)
    (= (capacidad F3) 1000)
    (= (capacidad F4) 1000)

    ; ;Gasto
    (= (gasto Av1) 20)
    (= (gasto Av2) 20)
    (= (gasto T1) 1)
    (= (gasto T2) 1)
    (= (gasto F1) 2)
    (= (gasto F2) 2)
    (= (gasto F3) 2)
    (= (gasto F4) 2)

)

(:goal (and
    (at P1 Ca3) (at P2 Ca3)
))

; (:metric minimize (total-time))
(:metric minimize (combustibletotal))
)