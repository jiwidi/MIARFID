(define (problem probtransporte)
    (:domain transporte)
    (:objects
        CIUDAD1 CIUDAD2 CIUDAD3 - ciudad
        Ca1 Ca2 - casa
        E1 E2 E3 - estacion
        A1 A2 - aeropuerto
        F1 F2 F3 - furgoneta
        T1 - tren
        Av1 Av2 - avion
        P1 P2 - paquete
        D1 D2 D3 - conductor
    )

    (:init
        ;ciudad 1
        (in CIUDAD1 Ca1)
        (in CIUDAD1 A1)
        (in CIUDAD1 E1)
        (in CIUDAD1 E2)

        (at F1 E1)
        (at D1 E1)

        (at F2 A1)
        (at Av1 A1)
        (at D2 A1)

        (at T1 E2)

        (at P1 Ca1)
        (at P2 Ca1)

        (empty F1)
        (empty F2)

        ;ciudad 2
        (in CIUDAD2 Ca2)
        (in CIUDAD2 A2)
        (in CIUDAD2 E3)

        (at F3 A2)
        (at Av2 A2)
        (at D3 A2)

        (empty F3)

    )

    (:goal
        (and
            (at P1 Ca2) (at P2 A2)
        )
    )

)