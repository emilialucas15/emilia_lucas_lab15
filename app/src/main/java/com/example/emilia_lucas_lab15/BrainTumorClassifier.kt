package com.example.emilia_lucas_lab15


import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs brain_tumor_classifier.tflite (trained in Colab).
 * Input:  [1, 160, 160, 3] float32, RGB, pixel values 0-255 (the model rescales internally).
 * Output: [1, 4] float32 probabilities, same order as labels.txt.
 */
class BrainTumorClassifier(context: Context) : AutoCloseable {

    private val interpreter = Interpreter(
        loadModel(context, MODEL_FILE),
        Interpreter.Options().setNumThreads(4)
    )

    val labels: List<String> = context.assets.open(LABELS_FILE)
        .bufferedReader()
        .readLines()
        .map { it.trim() }
        .filter { it.isNotEmpty() }

    init {
        val outputSize = interpreter.getOutputTensor(0).shape()[1]
        require(outputSize == labels.size) {
            "labels.txt has ${labels.size} labels but the model outputs $outputSize classes."
        }
    }

    /** Returns (label, probability) pairs sorted from most to least likely. */
    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        // Same size and bilinear filtering as in Colab
        val scaled = Bitmap.createScaledBitmap(argb, INPUT_SIZE, INPUT_SIZE, true)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat()) // R
            input.putFloat(((p shr 8) and 0xFF).toFloat())  // G
            input.putFloat((p and 0xFF).toFloat())          // B
        }
        input.rewind()

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)

        return labels.zip(output[0].toList()).sortedByDescending { it.second }
    }

    override fun close() = interpreter.close()

    private fun loadModel(context: Context, fileName: String): MappedByteBuffer =
        context.assets.openFd(fileName).use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }

    companion object {
        const val MODEL_FILE = "brain_tumor_classifier.tflite"
        const val LABELS_FILE = "labels.txt"
        const val INPUT_SIZE = 160
    }
}