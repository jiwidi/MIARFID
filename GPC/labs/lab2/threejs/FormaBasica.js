// Seminario GPC #2 Forma Basica.
// Dibujar formas basicas y un modelo importad
// Muestra el bucle tipico de inicializacion, escena y render


// Variables de consenso
// Motor, escena y camara
var renderer, scene, camara;

// Otras globales
var esferCubo, angulo = 0;

// Acciones
IntersectionObserver();
loadScene();
render();

function init() {
    // Configurar el motor de render y el canvas
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(new THREE.Color(0x0000AA));
}