import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:tflite_v2/tflite_v2.dart';
import 'package:image_picker/image_picker.dart';

// Constants
const Color pastelGreen = Color(0xFFC1E1C1);
const Color whiteColor = Colors.white;
const Color greenAccentColor = Colors.greenAccent;
const double buttonHeight = 60;
const double padding = 20;

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  File? _image;
  List? _output;
  final picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  Future<void> classifyImage(File image) async {
    var output = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 3,
      threshold: 0.0001,
      imageMean: 0,
      imageStd: 255,
    );
    setState(() => _output = output);
  }

  Future<void> loadModel() async {
    await Tflite.loadModel(
      model: 'android/app/src/main/assets/EfficientNetV2S_finetuned.tflite',
      labels: 'android/app/src/main/assets/labels.txt',
    );
  }

  Future<void> pickImage(ImageSource source) async {
    var image = await picker.pickImage(source: source, maxWidth: 384, maxHeight: 384);
    if (image != null) {
      setState(() => _image = File(image.path));
      classifyImage(_image!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(),
      body: _buildBody(context),
    );
  }

  AppBar _buildAppBar() {
    return AppBar(
      backgroundColor: greenAccentColor,
      centerTitle: true,
      title: Text('Dog Breed Classification', style: GoogleFonts.lato(color: whiteColor, fontWeight: FontWeight.w600, fontSize: 23)),
    );
  }

  Widget _buildBody(BuildContext context) {
    final double screenWidth = MediaQuery.of(context).size.width;
    const double bottomPadding = padding + buttonHeight * 2 + padding;

    return Container(
      color: pastelGreen,
      child: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.only(bottom: bottomPadding),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: padding),
              _buildImageSection(screenWidth),
              const SizedBox(height: padding),
              _buildButton('Take A Photo', () => pickImage(ImageSource.camera)),
              const SizedBox(height: padding),
              _buildButton('Pick From Gallery', () => pickImage(ImageSource.gallery)),
              if (_output != null) ..._buildResults(),
              const SizedBox(height: padding),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImageSection(double screenWidth) {
    return _image == null
        ? const Padding(
      padding: EdgeInsets.all(16.0),
      child: Text('No image selected', style: TextStyle(color: Colors.black, fontSize: 24.0)),
    )
        : SizedBox(
      width: screenWidth * 0.9,
      height: screenWidth * 0.9,
      child: Image.file(_image!, fit: BoxFit.cover),
    );
  }

  Widget _buildButton(String text, VoidCallback onPressed) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        foregroundColor: greenAccentColor,
        backgroundColor: whiteColor,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18.0)),
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 17),
      ),
      child: Text(text, style: GoogleFonts.lato()),
    );
  }

  List<Widget> _buildResults() {
    return _output!.map((result) => Padding(
      padding: const EdgeInsets.only(top: 16.0),
      child: Text(
        '${result['label']}: ${result['confidence'].toStringAsFixed(2)}',
        style: GoogleFonts.lato(color: Colors.black, fontSize: 20, fontWeight: FontWeight.w500),
      ),
    )).toList();
  }
}
