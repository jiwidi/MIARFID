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

(:requirements :strips :typing)

(:types paquete loc vehiculo conductor ciudad - object
        casa estacion aeropuerto - loc
        furgoneta tren avion - vehiculo
)

(:predicates ; Para definir en que ciudad esta cada cosa
             (in ?c - ciudad
                 ?x - (either paquete loc vehiculo conductor))
             
             ; Localizacion dentro de una ciudad
             (at ?x - (either paquete vehiculo conductor) ?l - loc)
             
             ; Vehiculo cargado con paquete p
             (loaded ?v - vehiculo ?p - paquete)

             ; Conductor encima de la furgoneta f
             (on ?c - conductor ?f - furgoneta)
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
        (in ?co ?a) ; El avion esta en la ciudad origen 
        (in ?cd ?d) ; Aeropuerto destino en ciudad destino - No se si es necesario
    )
    :effect (and 
        (at ?a ?d)
        (not (in ?co ?a)) ; El avion ya no esta en la ciudad origen
        (in ?cd ?a)       ; Ahora esta en la ciudad destino
    )
)

; Mover un tren entre dos estaciones
(:action mover_tren
    :parameters (?t - tren ?o - estacion ?d - estacion ?co - ciudad ?cd - ciudad)
    :precondition (and 
        (at ?t ?o)  ; Tren en estacion origen
        (in ?co ?t) ; Tren en ciudad origen
        (in ?cd ?d) ; Estacion destino en ciudad destino - No se si es necesario
    )
    :effect (and 
        (at ?t ?d)
        (not (in ?co ?t)) ; El tren ya no esta en la ciudad origen
        (in ?cd ?t)       ; Ahora esta en la ciudad destino
    )
)

; No se si así funcionarian bien los trenes o habria que hacer 2 acciones.
; Una para mover un tren en estaciones de la misma ciudad y otra para distintas ciudades
; (esto ultimo seguro que funcionaria bien)

)