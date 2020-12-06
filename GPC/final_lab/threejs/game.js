/**
 * Seminario GPC #4  Animación por simulación física.
 * Esferas en habitación cerrada con molinete central
 *
 * @requires three_r96.js, coordinates.js, orbitControls.js, cannon.js, tween.js, stats_r16.js
 * @author rvivo / http://personales.upv.es/rvivo
 * @date 2020
 */

// Globales convenidas por threejs
var renderer, scene, camera;
// Control de camara
var cameraControls;
// Monitor de recursos
var stats;
// Mundo fisico
var world, reloj;
var premio;
// Objetos
const nesferas = 20;
const nobstacurlos = 50;
var esferas = [];
var obstaculos = [];
var len_suelo = 80
initPhysicWorld();
initVisualWorld();
loadWorld();
render();
createLights();
var contador;
var puntos = 0;


function getRndInteger(min, max) {
	i = Math.floor(Math.random() * (max - min)) + min;
	while (i < 5 & i > 5) {
		i = Math.floor(Math.random() * (max - min)) + min;
	}
	return Math.floor(Math.random() * (max - min)) + min;
}

/**
 * Construye una bola con cuerpo y vista
 */
function pelota(radio, posicion, material) {
	var masa = 150;
	this.body = new CANNON.Body({
		mass: masa,
		material: material
	});
	var textureLoader = new THREE.TextureLoader();
	var map = textureLoader.load('./textures/ball.jpg');
	this.body.addShape(new CANNON.Sphere(radio));
	this.body.position.copy(posicion);
	this.visual = new THREE.Mesh(new THREE.SphereGeometry(radio, 32, 32),
		new THREE.MeshBasicMaterial({
			// wireframe: true,
			map: map,
			shading: THREE.FlatShading,
		}));
	this.visual.castShadow = true;
	this.visual.position.copy(this.body.position);
}
/**
 * Construye un obstaculo con cuerpo y vista
 */
function obstaculo(altura, posicion, material) {
	var masa = 100000;
	var textureLoader = new THREE.TextureLoader();
	var map = textureLoader.load('./textures/brick.png');
	this.body = new CANNON.Body({
		mass: masa,
		material: material
	});
	this.body.addShape(new CANNON.Box(new CANNON.Vec3(1, altura / 2, 1)));
	this.body.position.copy(posicion);
	var geom = new THREE.BoxGeometry(2, altura, 2, 10, 10, 10);
	var mat = new THREE.MeshBasicMaterial({
		side: THREE.DoubleSide,
		map: map,
		shading: THREE.FlatShading,
	});
	this.visual = new THREE.Mesh(geom, mat);
	this.visual.castShadow = true;
}
/**
 * Construye un diamante con cuerpo y vista, geometria de https://observablehq.com/@sxywu/three-js-exploration-shapes
 */
function premio(radio, posicion, materialFisico) {
	var textureLoader = new THREE.TextureLoader();
	var map = textureLoader.load('./textures/diamond.jpg');
	//Physical
	this.body = new CANNON.Body({
		mass: 0,
		material: materialFisico
	});
	this.body.addShape(new CANNON.Sphere(radio));
	this.body.position.copy(posicion);
	// create just one triangle
	var vertices = [
		-1, 1, 1, // 0: left top front
		-1, -1, 1, // 1: left bottom front
		1, -1, 1, // 2: right bottom front
		1, 1, 1, // 3: right top front
		1, -1, -1, // 4: right bottom back
		1, 1, -1, // 5: right top back
		-1, -1, -1, // 6: left bottom back
		-1, 1, -1, // 7: left top back
		0, 1, 0, // 8: top middle
		0, -1, 0, // 9: bottom middle
	]
	var faces = [
		0, 1, 2, // front 1
		0, 2, 3, // front 2
		3, 2, 4, // right 1
		3, 4, 5, // right 2
		5, 4, 6, // back 1
		5, 6, 7, // back 2
		7, 6, 1, // left 1
		7, 1, 0, // left 2
		8, 0, 3, // top front
		8, 3, 5, // top right
		8, 5, 7, // top back
		8, 7, 0, // top left
		9, 2, 1, // bottom front
		9, 4, 2, // bottom right
		9, 6, 4, // bottom back
		9, 1, 6, // bottom left
	]
	var geometry = new THREE.PolyhedronGeometry(vertices, faces, 30, 0);
	var material = new THREE.MeshNormalMaterial({
		// side: THREE.DoubleSide,
		// map: map,
		// shading: THREE.FlatShading,
	});
	var mesh = new THREE.Mesh(geometry, material);
	mesh.castShadow = true;
	// scale the mesh
	mesh.scale.set(0.03, 0.06, 0.03)
	this.visual = mesh;
}
/**
 * Crea las luces
 */
function createLights() {

	hemisphereLight = new THREE.HemisphereLight(0xaaaaaa, 0x000000, .9)
	shadowLight = new THREE.DirectionalLight(0xffffff, .9);
	shadowLight.position.set(len_suelo, len_suelo, 50);
	shadowLight.castShadow = true;
	shadowLight.shadow.camera.left = -400;
	shadowLight.shadow.camera.right = 400;
	shadowLight.shadow.camera.top = 400;
	shadowLight.shadow.camera.bottom = -400;
	shadowLight.shadow.camera.near = 1;
	shadowLight.shadow.camera.far = 1000;
	shadowLight.shadow.mapSize.width = 2048;
	shadowLight.shadow.mapSize.height = 2048;
	shadowLight.shadowMapWidth = 2024; // default is 512
	shadowLight.shadowMapHeight = 2024;
	shadowLight.shadowCameraVisible = true;
	shadowLight.angle = Math.PI / 8.0;

	scene.add(hemisphereLight);
	scene.add(shadowLight);
}

/**
 * Inicializa el mundo fisico con un
 * suelo y cuatro paredes de altura infinita
 */
function initPhysicWorld() {
	// Mundo
	world = new CANNON.World();
	world.gravity.set(0, -9.8, 0);
	///world.broadphase = new CANNON.NaiveBroadphase();
	world.solver.iterations = 10;

	// Material y comportamiento
	var groundMaterial = new CANNON.Material("groundMaterial");
	var materialEsfera = new CANNON.Material("sphereMaterial");
	var obstacleMaterial = new CANNON.Material("obstacleMaterial");
	world.addMaterial(materialEsfera);
	world.addMaterial(groundMaterial);
	world.addMaterial(obstacleMaterial);
	// -existe un defaultContactMaterial con valores de restitucion y friccion por defecto
	// -en caso que el material tenga su friccion y restitucion positivas, estas prevalecen
	var sphereGroundContactMaterial = new CANNON.ContactMaterial(groundMaterial, materialEsfera, {
		friction: 0.3,
		restitution: 0.7
	});
	var sphereObstacleContactMaterial = new CANNON.ContactMaterial(materialEsfera, obstacleMaterial, {
		friction: 0.3,
		restitution: 0.7
	});
	var obstacleGroundContactMaterial = new CANNON.ContactMaterial(obstacleMaterial, groundMaterial, {
		friction: 0.3,
		restitution: 0.7
	});
	world.addContactMaterial(sphereGroundContactMaterial);
	world.addContactMaterial(sphereObstacleContactMaterial);
	world.addContactMaterial(obstacleGroundContactMaterial);

	// Suelo
	var groundShape = new CANNON.Plane();
	var ground = new CANNON.Body({
		mass: 0,
		material: groundMaterial
	});
	ground.addShape(groundShape);
	ground.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
	world.addBody(ground);

	// Paredes
	var backWall = new CANNON.Body({
		mass: 0,
		material: groundMaterial
	});
	backWall.addShape(new CANNON.Plane());
	backWall.position.z = -1;
	world.addBody(backWall);
	var frontWall = new CANNON.Body({
		mass: 0,
		material: groundMaterial
	});
	frontWall.addShape(new CANNON.Plane());
	frontWall.quaternion.setFromEuler(0, Math.PI, 0, 'XYZ');
	frontWall.position.z = 1;
	world.addBody(frontWall);
	var leftWall = new CANNON.Body({
		mass: 0,
		material: groundMaterial
	});
	leftWall.addShape(new CANNON.Plane());
	leftWall.position.x = -2;
	leftWall.quaternion.setFromEuler(0, Math.PI / 2, 0, 'XYZ');
	world.addBody(leftWall);
	var rightWall = new CANNON.Body({
		mass: 0,
		material: groundMaterial
	});
	rightWall.addShape(new CANNON.Plane());
	rightWall.position.x = len_suelo - 2;
	rightWall.quaternion.setFromEuler(0, -Math.PI / 2, 0, 'XYZ');
	world.addBody(rightWall);
}

/**
 * Inicializa la escena visual, crea la habitacion, carretera y texto de instrucciones
 */
function initVisualWorld() {
	// Inicializar el motor de render
	renderer = new THREE.WebGLRenderer();
	renderer.setSize(window.innerWidth, window.innerHeight);
	// renderer.setClearColor(new THREE.Color(0x000000));
	renderer.setClearColor(new THREE.Color(0xd8d0d1), 1);
	renderer.shadowMapEnabled = true
	renderer.shadowMapType = THREE.PCFSoftShadowMap;
	document.getElementById('container').appendChild(renderer.domElement);

	// Crear el grafo de escena
	scene = new THREE.Scene();

	var texture = new THREE.TextureLoader().load("textures/sky.jpg");
	scene.background = texture;

	// Reloj
	reloj = new THREE.Clock();
	reloj.start();

	// Crear y situar la camara
	var aspectRatio = window.innerWidth / window.innerHeight;
	camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 100);
	camera.position.set(0, 5, 7);
	camera.lookAt(new THREE.Vector3(0, 0, 0));

	// STATS --> stats.update() en update()
	stats = new Stats();
	stats.showPanel(0); // FPS inicialmente. Picar para cambiar panel.
	document.getElementById('container').appendChild(stats.domElement);

	// Callbacks
	window.addEventListener('resize', updateAspectRatio);

	//Carretera
	var textureLoader = new THREE.TextureLoader();
	var map = textureLoader.load('./textures/road.jpg');
	var geometry = new THREE.PlaneGeometry(len_suelo + 1, 2, 3200);
	geometry.applyMatrix(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
	var material = new THREE.MeshPhongMaterial({
		side: THREE.DoubleSide,
		map: map
	});
	var plane = new THREE.Mesh(geometry, material);
	plane.position.x = len_suelo / 2 - 2
	plane.receiveShadow = true;
	scene.add(plane);

	//Suelo visual
	var textureLoader = new THREE.TextureLoader();
	var map = textureLoader.load('./textures/asphalt.jpg');
	var geometry = new THREE.PlaneGeometry(len_suelo * 2, len_suelo * 2, 512);
	geometry.applyMatrix(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
	var material = new THREE.MeshPhongMaterial({
		side: THREE.DoubleSide,
		map: map
	});
	var plane = new THREE.Mesh(geometry, material);
	plane.position.y -= 0.1
	plane.receiveShadow = true;
	scene.add(plane);

	// Habitacion
	var path = "textures/"
	var paredes = [path + 'pond/posx.jpg', path + 'pond/negx.jpg',
		path + 'pond/posy.jpg', path + 'pond/negy.jpg',
		path + 'pond/posz.jpg', path + 'pond/negz.jpg'
	];
	var mapaEntorno = new THREE.CubeTextureLoader().load(paredes);

	var shader = THREE.ShaderLib.cube;
	shader.uniforms.tCube.value = mapaEntorno;

	var matparedes = new THREE.ShaderMaterial({
		fragmentShader: shader.fragmentShader,
		vertexShader: shader.vertexShader,
		uniforms: shader.uniforms,
		dephtWrite: false,
		side: THREE.BackSide
	});

	var habitacion = new THREE.Mesh(new THREE.CubeGeometry(len_suelo * 2, len_suelo * 2, len_suelo * 2), matparedes);
	scene.add(habitacion);

	//Texto
	var loader = new THREE.FontLoader();
	loader.load('fonts/helvetiker_regular.typeface.json', function (font) {
		var textGeo = new THREE.TextGeometry("Find the diamond at the \n end of the road", {
			font: font,
			size: 2,
			height: 0.5,
			// curveSegments: 12,
			// bevelThickness: 2,
			// bevelSize: 5,
			// bevelEnabled: true

		});
		var textMaterial = new THREE.MeshPhongMaterial({
			color: 0xff0000
		});
		var mesh = new THREE.Mesh(textGeo, textMaterial);
		mesh.position.set(-6, 8, -15);
		mesh.castShadow = true;
		scene.add(mesh);
	});
	loader.load('fonts/helvetiker_regular.typeface.json', function (font) {
		var textGeo = new THREE.TextGeometry("Move with WASD", {
			font: font,
			size: 2,
			height: 0.5,
			// curveSegments: 12,
			// bevelThickness: 2,
			// bevelSize: 5,
			// bevelEnabled: true

		});
		var textMaterial = new THREE.MeshNormalMaterial();
		var mesh = new THREE.Mesh(textGeo, textMaterial);
		mesh.position.set(-6, 2, -15);
		mesh.castShadow = true;
		scene.add(mesh);
	});
	loader.load('fonts/helvetiker_regular.typeface.json', function (font) {
		var textGeo = new THREE.TextGeometry("Points: " + puntos, {
			font: font,
			size: 2,
			height: 0.5,
		});
		var textMaterial = new THREE.MeshNormalMaterial();
		contador = new THREE.Mesh(textGeo, textMaterial);
		contador.name = "puntos"
		contador.position.set(20, 2, -15);
		contador.castShadow = true;
		scene.add(contador);
	});
}

/**
 * Carga los objetos(pelota, obstaculos, diamante) es el mundo físico y visual
 */
function loadWorld() {
	// Genera las esferas
	var materialEsfera;
	var materialObstaculo;
	for (i = 0; i < world.materials.length; i++) {
		if (world.materials[i].name === "sphereMaterial") materialEsfera = world.materials[i];
		if (world.materials[i].name === "obstacleMaterial") materialObstaculo = world.materials[i];
	}
	pelota_jugador = new pelota(1 / 2, new CANNON.Vec3(-1, 3, 0), materialEsfera);
	world.addBody(pelota_jugador.body);
	scene.add(pelota_jugador.visual);

	for (var i = 4; i < len_suelo - 5; i += 2) {
		var r = getRndInteger(0, 2);
		if (r > 0) {
			var altura = getRndInteger(1, 8);
			// for (var j = 0; j < n_obs; j++) {
			var obs = new obstaculo(altura, new CANNON.Vec3(i, 0, -0.5), materialObstaculo);
			world.addBody(obs.body);
			scene.add(obs.visual);
			obstaculos.push(obs);
			// }
		}
	};
	reward = new premio(1, new CANNON.Vec3(len_suelo - 3, 2, 0), materialObstaculo);
	world.addBody(reward.body);
	scene.add(reward.visual);
	scene.add(new THREE.AxisHelper(5));

	var giro = new TWEEN.Tween(reward.visual.rotation).to({
		x: 0,
		y: 2 * Math.PI,
		z: 0
	}, 10000);
	giro.repeat(Infinity);
	giro.start();

	//Controller
	window.addEventListener('keydown', function movekey(event) {
		if (event.keyCode == 39 || event.keyCode == 68) {
			pelota_jugador.body.velocity.x = 0;
			pelota_jugador.body.applyImpulse(new CANNON.Vec3(+800, 0, 0), pelota_jugador.body.position)
			pelota_jugador.visual.position.copy(pelota_jugador.body.position);

			// Luces
			// shadowLight.position.x += 10;
		} else if ((event.keyCode == 38 || event.keyCode == 87)) {
			pelota_jugador.body.applyImpulse(new CANNON.Vec3(0, 0, -300), pelota_jugador.body.position)
			pelota_jugador.visual.position.copy(pelota_jugador.body.position);
			console.log("w")
		} else if ((event.keyCode == 37 || event.keyCode == 65)) {
			pelota_jugador.body.velocity.x = 0;
			pelota_jugador.body.applyImpulse(new CANNON.Vec3(-800, 0, 0), pelota_jugador.body.position)
			pelota_jugador.visual.position.copy(pelota_jugador.body.position);
			// shadowLight.position.x -= 10;
			console.log("a")

		} else if ((event.keyCode == 40 || event.keyCode == 83)) {
			pelota_jugador.body.applyImpulse(new CANNON.Vec3(0, 0, 300), pelota_jugador.body.position)
			pelota_jugador.visual.position.copy(pelota_jugador.body.position);
			console.log("s")
		} else if ((event.keyCode == 32)) {
			if (pelota_jugador.body.position.y < 7) {
				pelota_jugador.body.velocity.y = 0
				pelota_jugador.body.applyImpulse(new CANNON.Vec3(0, 1400, 0), pelota_jugador.body.position)
				pelota_jugador.visual.position.copy(pelota_jugador.body.position);
				console.log("space")
			}

		}
	}, false);
}

/**
 * Isotropía frente a redimension del canvas
 */
function updateAspectRatio() {
	renderer.setSize(window.innerWidth, window.innerHeight);
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
}
//Resets the game
function reset() {
	obstaculos = [];

	for (var i = 0; i < obstaculos.length; i++) {
		world.removeBody(obstaculos[i].body)
		scene.rem
	};
	loadWorld();
}

/**
 * Actualizacion segun pasa el tiempo
 */
function update() {
	var segundos = reloj.getDelta(); // tiempo en segundos que ha pasado
	world.step(segundos); // recalcula el mundo tras ese tiempo

	//Actualizamos visual pelota
	pelota_jugador.visual.position.copy(pelota_jugador.body.position);
	pelota_jugador.visual.quaternion.copy(pelota_jugador.body.quaternion);

	//Actualizamos visual pelota
	reward.visual.position.copy(reward.body.position);
	reward.visual.quaternion.copy(reward.body.quaternion);



	for (var i = 0; i < obstaculos.length; i++) {
		obstaculos[i].visual.position.copy(obstaculos[i].body.position);
		obstaculos[i].visual.quaternion.copy(obstaculos[i].body.quaternion);
	};
	camera.position.x = pelota_jugador.body.position.x
	camera.position.y = pelota_jugador.body.position.y + 10
	camera.position.z = pelota_jugador.body.position.z + 10

	// Actualiza el monitor
	stats.update();

	// Actualiza el movimeinto del molinete
	TWEEN.update();

	// Checkeamos condicion de ganar:
	if (pelota_jugador.body.position.x > len_suelo - 4 & pelota_jugador.body.position.y < 5) {

		pelota_jugador.body.position.y = 1
		pelota_jugador.body.position.x = 1
		pelota_jugador.body.velocity.y = 0
		puntos += 1
		var selectedObject = scene.getObjectByName(contador.name);
		scene.remove(selectedObject);
		var loader = new THREE.FontLoader();
		loader.load('fonts/helvetiker_regular.typeface.json', function (font) {
			var textGeo = new THREE.TextGeometry("Points: " + puntos, {
				font: font,
				size: 2,
				height: 0.5,

			});
			var textMaterial = new THREE.MeshNormalMaterial();
			contador = new THREE.Mesh(textGeo, textMaterial);
			contador.position.set(20, 2, -15);
			contador.castShadow = true;
			scene.add(contador);
		});
		// reset();
	}
}

/**
 * Update & render
 */
function render() {
	requestAnimationFrame(render);
	update();
	renderer.render(scene, camera);
}