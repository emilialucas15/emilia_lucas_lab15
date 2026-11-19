package com.example.emilia_lucas_lab15

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {

    // El clasificador se crea una sola vez al abrir la app, porque cargar el modelo
    // tarda un poco; así no se vuelve a cargar con cada imagen.
    private lateinit var classifier: BrainTumorClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        classifier = BrainTumorClassifier(this) // carga el .tflite y labels.txt desde assets
        setContent {
            // Tema Material por defecto (botón morado y fondo claro, como en el codelab)
            MaterialTheme {
                LabelImageScreen(classifier)
            }
        }
    }

    override fun onDestroy() {
        classifier.close() // libera la memoria del modelo al cerrar la app
        super.onDestroy()
    }
}

@Composable
fun LabelImageScreen(classifier: BrainTumorClassifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Estado de la pantalla: cuando cambia, Compose redibuja la interfaz automáticamente
    var image by remember { mutableStateOf<Bitmap?>(null) }   // imagen elegida
    var outputText by remember { mutableStateOf("") }          // resultado "etiqueta : confianza"
    var busy by remember { mutableStateOf(false) }             // true mientras clasifica

    // Abre el explorador de archivos del teléfono para elegir una imagen.
    // Usamos OpenDocument porque así se ve la carpeta Descargas,
    // donde quedan las imágenes que arrastramos al emulador.
    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult // el usuario canceló
        scope.launch {
            busy = true
            try {
                // La lectura y la predicción se hacen en segundo plano (Dispatchers.Default)
                // para que la pantalla no se congele mientras el modelo trabaja.
                val (bmp, results) = withContext(Dispatchers.Default) {
                    val bmp = context.contentResolver.openInputStream(uri)
                        .use { BitmapFactory.decodeStream(it) }
                        ?: throw IllegalArgumentException("This file could not be read as an image.")
                    bmp to classifier.classify(bmp) // <- aquí se ejecuta el modelo .tflite
                }
                image = bmp
                // Igual que en el codelab: una línea por etiqueta con el formato "label : confidence".
                // La lista ya viene ordenada, así que la primera línea es la predicción.
                outputText = results.joinToString("\n") { (label, confidence) ->
                    "$label : $confidence"
                }
            } catch (e: Exception) {
                outputText = e.message ?: "The image could not be labeled."
            } finally {
                busy = false
            }
        }
    }

    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Imagen elegida, arriba y centrada (como en el codelab)
            image?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Selected MRI image",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.size(160.dp)
                )
            }

            // Botón que elige la imagen y la clasifica.
            // Se desactiva mientras clasifica para no lanzar dos predicciones a la vez.
            Button(
                onClick = { picker.launch(arrayOf("image/*")) }, // solo permite archivos de imagen
                enabled = !busy
            ) {
                Text("Label Image")
            }

            // Resultado en texto, alineado a la izquierda
            Text(
                text = outputText,
                modifier = Modifier.fillMaxWidth(),
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}