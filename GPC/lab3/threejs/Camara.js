/**
 *	Seminario GPC #3. Camara
 *	Manejar diferentes camaras, marcos y picking
 *
 */

// Variables imprescindibles
var renderer, scene, camera;

// Variables globales
var esferacubo, cubo, angulo = 0;
var l = b = -4;
var r = t = -l;
var cameraControls;
var alzado, planta, perfil;

// Acciones
init();
loadScene();
render();

function setCameras(ar) {
    // Construye las camaras planta, alzado, perfil y perspectiva
    var origen = new THREE.Vector3(0, 0, 0);

    if (ar > 1)
        var camaraOrtografica = new THREE.OrthographicCamera(l * ar, r * ar, t, b, -20, 20);
    else
        var camaraOrtografica = new THREE.OrthographicCamera(l, r, t / ar, b / ar, -20, 20);

    // Camaras ortograficas
    alzado = camaraOrtografica.clone();
    alzado.position.set(0, 0, 4);
    alzado.lookAt(origen);
    perfil = camaraOrtografica.clone();
    perfil.position.set(4, 0, 0);
    perfil.lookAt(origen);
    planta = camaraOrtografica.clone();
    planta.position.set(0, 4, 0);
    planta.lookAt(origen);
    planta.up = new THREE.Vector3(0, 0, -1);

    // Camara perspectiva
    var camaraPerspectiva = new THREE.PerspectiveCamera(50, ar, 0.1, 50);
    camaraPerspectiva.position.set(1, 2, 10);
    camaraPerspectiva.lookAt(origen);

    camera = camaraPerspectiva.clone();

    scene.add(alzado);
    scene.add(planta);
    scene.add(perfil);
    scene.add(camera);

}

function init() {
    // Crear el motor, la escena y la camara

    // Motor de render
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(new THREE.Color(0x0000AA));
    renderer.autoClear = false;
    document.getElementById('container').appendChild(renderer.domElement);

    // Escena
    scene = new THREE.Scene();

    // Camara
    var ar = window.innerWidth / window.innerHeight;
    setCameras(ar);

    // Controlador de camara
    cameraControls = new THREE.OrbitControls(camera, renderer.domElement);
    cameraControls.target.set(0, 0, 0);
    cameraControls.noKeys = true;

    // Captura de eventos
    window.addEventListener('resize', updateAspectRatio);
    renderer.domElement.addEventListener('dblclick', rotate);
}

function loadScene() {
    // Cargar la escena con objetos

    // Materiales
    var material = new THREE.MeshBasicMaterial({
        color: 'yellow',
        wireframe: true
    });

    // Geometrias
    var geocubo = new THREE.BoxGeometry(2, 2, 2);
    var geoesfera = new THREE.SphereGeometry(1, 30, 30);

    // Objetos
    cubo = new THREE.Mesh(geocubo, material);
    cubo.position.x = -1;

    var esfera = new THREE.Mesh(geoesfera, material);
    esfera.position.x = 1;

    esferacubo = new THREE.Object3D();
    esferacubo.position.y = 1;

    // Modelo importado
    var loader = new THREE.ObjectLoader();
    loader.load('models/soldado/soldado.json',
        function (obj) {
            obj.position.y = 1;
            cubo.add(obj);
        });

    // Construir la escena
    esferacubo.add(cubo);
    esferacubo.add(esfera);
    scene.add(esferacubo);
    //cubo.add(new THREE.AxisHelper(1));
    //scene.add( new THREE.AxisHelper(3) );

}

function rotate(event) {
    // Gira el objeto senyalado 45 grados

    var x = event.clientX;
    var y = event.clientY;

    // Transformacion a cuadrado de 2x2
    x = (x / window.innerWidth) * 2 - 1;
    y = -(y / window.innerHeight) * 2 + 1;

    console.log(x + ',' + y);

    var rayo = new THREE.Raycaster();
    rayo.setFromCamera(new THREE.Vector2(x, y), camera);

    var interseccion = rayo.intersectObjects(scene.children, true);
    console.log('objs: ' + scene.children.length);
    console.log('int: ' + interseccion.length);
    if (interseccion.length > 0) {

        interseccion[0].object.rotation.y += Math.PI / 4;
    }
}

function updateAspectRatio() {
    // Renueva la relacion de aspecto de la camara

    // Ajustar el tamaño del canvas
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Razon de aspecto
    var ar = window.innerWidth / window.innerHeight;

    /* Camara ortografica
    if( ar > 1 ){
    	camera.left = -4*ar;
    	camera.right = 4*ar;
    	camera.bottom = -4;
    	camera.top = 4;
    }
    else {
    	camera.top = 4/ar;
    	camera.bottom = -4/ar;
    	camera.left = -4;
    	camera.right = 4;
    }
    */

    // Camara perspectiva
    camera.aspect = ar;

    camera.updateProjectionMatrix();
}

function update() {
    // Cambios entre frames

}

function render() {
    // Dibujar cada frame
    requestAnimationFrame(render);

    update();

    renderer.clear();

    // Para cada render debo indicar el viewport

    renderer.setViewport(window.innerWidth / 2, 0,
        window.innerWidth / 2, window.innerHeight / 2);
    renderer.render(scene, perfil);
    renderer.setViewport(0, 0,
        window.innerWidth / 2, window.innerHeight / 2);
    renderer.render(scene, alzado);
    renderer.setViewport(0, window.innerHeight / 2,
        window.innerWidth / 2, window.innerHeight / 2);
    renderer.render(scene, planta);
    renderer.setViewport(window.innerWidth / 2, window.innerHeight / 2,
        window.innerWidth / 2, window.innerHeight / 2);
    renderer.render(scene, camera);
}