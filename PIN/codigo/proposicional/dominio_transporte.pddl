;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; PROBLEMA TRANSPORTE                                      ;;;
;;; Ejercicio 1: Dominio Proposicional                       ;;;
;;;                                                          ;;;
;;; Planificacion Inteligente @ MIARFID, UPV                 ;;;
;;;                                                          ;;;
;;; Autores:                                                 ;;;
;;;         Jaime Ferrando Huertas                           ;;;
;;;         Javier Martínez Bernia                           ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain transporte)

(:requirements :strips :typing :equality)

(:types paquete loc vehiculo conductor ciudad - object
        casa estacion aeropuerto - loc
        furgoneta tren avion - vehiculo
)

(:predicates ; Para definir en que ciudad esta cada cosa
             (in ?c - ciudad
                 ?x - loc)

             ; Localizacion dentro de una ciudad
             (at ?x - (either paquete vehiculo conductor) ?l - loc)

             ; Vehiculo cargado con paquete p
             (loaded ?v - vehiculo ?p - paquete)

             ; Conductor encima de la furgoneta f
             (on ?c - conductor ?f - furgoneta)

             ; Furgoneta sin conductor
             (empty ?f - furgoneta)
)

; ACCIONES

; Cargar un paquete en un vehiculo
(:action load
    :parameters (?v - vehiculo ?l - loc ?p - paquete)
    :precondition (and
        (at ?v ?l)
        (at ?p ?l)
    )
    :effect (and
        (not (at ?p ?l)) (loaded ?v ?p)
    )
)

; Descargar un paquete de un vehiculo
(:action unload
    :parameters (?v - vehiculo ?l - loc ?p - paquete)
    :precondition (and
        (at ?v ?l)
        (loaded ?v ?p)
    )
    :effect (and
        (at ?p ?l) (not (loaded ?v ?p))
    )
)

; Mover un avion entre dos aeropuertos
(:action mover_avion
    :parameters (?a - avion ?o - aeropuerto ?d - aeropuerto ?co - ciudad ?cd - ciudad)
    :precondition (and
        (at ?a ?o)  ; El avion esta en el aeropuerto origen
        (not (= ?co ?cd))
        (in ?co ?o) ; Aeropuerto origen en ciudad origen
        (in ?cd ?d) ; Aeropuerto destino en ciudad destino
    )
    :effect (and
        (not(at ?a ?o))
        (at ?a ?d)
    )
)

; Mover un tren entre dos estaciones
(:action mover_tren
    :parameters (?t - tren ?o - estacion ?d - estacion)
    :precondition (and
        (at ?t ?o)  ; Tren en estacion origen
        (not (= ?o ?d)) ; Las estaciones son distintas
    )
    :effect (and
        (not (at ?t ?o))
        (at ?t ?d)
    )
)

; Mover una furgoneta
(:action mover_furgoneta
    :parameters (?f - furgoneta ?c - conductor ?o - loc ?d - loc ?ci - ciudad)
    :precondition (and
        (on ?c ?f) ; Conductor subido en furgoneta
        (at ?f ?o) ; Furgoneta en origen
        (in ?ci ?o) ; Las dos localizaciones son de la misma ciudad
        (in ?ci ?d)
    )
    :effect (and
        (not (at ?f ?o))
        (at ?f ?d) ; Furgoneta en posicion destino
    )
)

; Mover un conductor por una ciudad
(:action mover_conductor
    :parameters (?c - conductor ?o - loc ?d - loc ?ci - ciudad)
    :precondition (and
        (at ?c ?o) ; Solo existira el predicado at para un conductor cuando esté a pie
        (in ?ci ?o) ; Las dos localizaciones son de la misma ciudad
        (in ?ci ?d)
    )
    :effect (and
        (not (at ?c ?o))
        (at ?c ?d) ; Conductor en localizacion destino
    )
)

; Subir un conductor a una furgoneta
(:action subir_conductor
    :parameters (?f - furgoneta ?c - conductor ?l - loc)
    :precondition (and
        (at ?f ?l)
        (at ?c ?l)
        (empty ?f) ; Furgoneta libre (sin conductor)
    )
    :effect (and
        (not (at ?c ?l)) ; Para que el conductor no pueda desplazarse a pie (si no se baja)
        (on ?c ?f)
        (not (empty ?f))
    )
)

; Bajar un conductor de una furgoneta
(:action bajar_conductor
    :parameters (?f - furgoneta ?c - conductor ?l - loc)
    :precondition (and
        (at ?f ?l)
        (on ?c ?f)
    )
    :effect (and
        (not (on ?c ?f))
        (at ?c ?l)
        (empty ?f)
    )
)

)