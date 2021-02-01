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
        P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19 P20 P21 P22 P23 P24 P25 P26 P27 P28 P29 P30 P31 P32 P33 P34 P35 P36 P37 P38 P39 P40 - paquete
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
        (at D1 E2)
        (empty F1)
        (empty F2)

        (in CIUDAD2 Ca2)
        (in CIUDAD2 A2)
        (in CIUDAD2 E3)
        (at F3 A2)
        (at Av2 A2)
        (at D2 Ca2)
        (empty F3)

        (in CIUDAD3 Ca3)
        (in CIUDAD3 E4)
        (at F4 Ca3)
        (at T2 E4)
        (at D3 Ca3)
        (empty F4)

        ;Paquetes
        (at P1 Ca1)
        (at P2 Ca2)
        (at P3 E1)
        (at P4 E2)
        (at P5 A1)
        (at P6 A2)
        (at P7 E4)
        (at P8 Ca3)
        (at P9 E2)
        (at P10 A2)
        (at P11 Ca2)
        (at P12 Ca3)
        (at P13 E2)
        (at P14 E1)
        (at P15 A2)
        (at P16 A2)
        (at P17 E1)
        (at P18 Ca3)
        (at P19 E2)
        (at P20 Ca1)
        (at P21 Ca2)
        (at P22 Ca1)
        (at P23 A2)
        (at P24 A1)
        (at P25 A1)
        (at P26 Ca2)
        (at P27 E2)
        (at P28 Ca1)
        (at P29 E2)
        (at P30 Ca2)
        (at P31 Ca2)
        (at P32 Ca3)
        (at P33 E2)
        (at P34 E1)
        (at P35 A2)
        (at P36 A2)
        (at P37 E1)
        (at P38 Ca3)
        (at P39 E2)
        (at P40 Ca1)

        ; Temporal
        (= (peso P1) 10)
        (= (peso P2) 15)
        (= (peso P3) 20)
        (= (peso P4) 10)
        (= (peso P5) 15)
        (= (peso P6) 20)
        (= (peso P7) 10)
        (= (peso P8) 15)
        (= (peso P9) 20)
        (= (peso P10) 20)

        (= (peso P11) 10)
        (= (peso P12) 15)
        (= (peso P13) 20)
        (= (peso P14) 10)
        (= (peso P15) 15)
        (= (peso P16) 20)
        (= (peso P17) 10)
        (= (peso P18) 15)
        (= (peso P19) 20)
        (= (peso P20) 20)

        (= (peso P21) 10)
        (= (peso P22) 15)
        (= (peso P23) 20)
        (= (peso P24) 10)
        (= (peso P25) 15)
        (= (peso P26) 20)
        (= (peso P27) 10)
        (= (peso P28) 15)
        (= (peso P29) 20)
        (= (peso P30) 20)

        (= (peso P31) 10)
        (= (peso P32) 15)
        (= (peso P33) 20)
        (= (peso P34) 10)
        (= (peso P35) 15)
        (= (peso P36) 20)
        (= (peso P37) 10)
        (= (peso P38) 15)
        (= (peso P39) 20)
        (= (peso P40) 20)

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

        ; Ciudad 2
        (= (distancia A2 E3) 150)
        (= (distancia E3 A2) 150)
        (= (distancia A2 Ca2) 80)
        (= (distancia Ca2 A2) 80)
        (= (distancia Ca2 E3) 150)
        (= (distancia E3 Ca2) 150)

        ; Ciudad 3
        (= (distancia Ca3 E4) 70)
        (= (distancia E4 Ca3) 70)

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

        ;; Distancias aeropuertos
        (= (distancia A1 A2) 200)
        (= (distancia A2 A1) 200)
        ;; Distancias estaciones
        ;(= (distancia E1 E2) 60) (= (distancia E2 E1) 60) No se pone porque ya esta puesta arriba
        (= (distancia E1 E3) 300)
        (= (distancia E3 E1) 300)
        (= (distancia E1 E4) 250)
        (= (distancia E4 E1) 250)

        (= (distancia E2 E3) 300)
        (= (distancia E3 E2) 300)
        (= (distancia E2 E4) 200)
        (= (distancia E4 E2) 200)

        (= (distancia E3 E4) 250)
        (= (distancia E4 E3) 250)

    )

    (:goal
        (and
            (at P1 Ca3) (at P2 Ca3) (at P3 Ca3) (at P4 Ca3)
            (at P5 Ca2) (at P6 Ca1) (at P7 Ca2) (at P8 Ca1)
            (at P9 A2) (at P10 A1)
            (at P11 Ca3) (at P12 Ca3) (at P13 Ca3) (at P14 Ca3)
            (at P15 Ca2) (at P16 Ca1) (at P17 Ca2) (at P18 Ca1)
            (at P19 A2) (at P20 A1)
            (at P21 Ca3) (at P22 Ca3) (at P23 Ca3) (at P24 Ca3)
            (at P25 Ca2) (at P26 Ca1) (at P27 Ca2) (at P28 Ca1)
            (at P29 A2) (at P30 A1)
            (at P31 Ca3) (at P32 Ca3) (at P33 Ca3) (at P34 Ca3)
            (at P35 Ca2) (at P36 Ca1) (at P37 Ca2) (at P38 Ca1)
            (at P39 A2) (at P40 A1)
        )
    )

    (:metric minimize
        (combustibletotal)
    )

)