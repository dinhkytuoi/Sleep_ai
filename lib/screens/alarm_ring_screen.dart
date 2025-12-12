import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';

class AlarmRingScreen extends StatefulWidget {
  final String? toneAsset; // ✅ Lưu lại tham số truyền vào

  const AlarmRingScreen({super.key, this.toneAsset});

  @override
  State<AlarmRingScreen> createState() => _AlarmRingScreenState();
}

class _AlarmRingScreenState extends State<AlarmRingScreen> {
  final AudioPlayer _player = AudioPlayer();
  bool _isPlaying = false;
  String? _soundName;

  @override
  void initState() {
    super.initState();

    // Đợi context sẵn sàng để lấy arguments
    Future.microtask(() {
      final arg = ModalRoute.of(context)?.settings.arguments;

      if (arg is String && arg.isNotEmpty) {
        _soundName = arg;
      } else {
        _soundName = widget.toneAsset;
      }

      _startAlarm();
    });
  }

  Future<void> _startAlarm() async {
    try {
      await _player.setReleaseMode(ReleaseMode.loop);

      // ✅ Nếu không có âm thanh nào được chỉ định, phát mặc định
      String sourcePath = (_soundName != null && _soundName!.isNotEmpty)
          ? _soundName!
          : 'assets/audio/drizzling.mp3';

      // ✅ Xác định loại nguồn âm thanh
      if (sourcePath.startsWith('assets/')) {
        await _player.play(AssetSource(sourcePath.replaceFirst('assets/', '')));
      } else if (sourcePath.startsWith('audio/')) {
        await _player.play(AssetSource(sourcePath));
      } else if (sourcePath.startsWith('/')) {
        await _player.play(DeviceFileSource(sourcePath));
      } else {
        await _player.play(AssetSource('audio/$sourcePath.mp3'));
      }

      if (mounted) setState(() => _isPlaying = true);
      debugPrint('🔊 Playing alarm sound: $sourcePath');
    } catch (e) {
      debugPrint('❌ Error starting alarm audio: $e');
    }
  }

  Future<void> _stopAlarm() async {
    await _player.stop();
    if (mounted) Navigator.pop(context);
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black.withOpacity(0.9),
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.alarm_rounded,
                  color: Colors.redAccent, size: 120),
              const SizedBox(height: 30),
              const Text(
                "⏰ Alarm is ringing!",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 60),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.redAccent,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 48, vertical: 18),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                onPressed: _stopAlarm,
                child: const Text(
                  "Stop Alarm!",
                  style: TextStyle(fontSize: 20, color: Colors.white),
                ),
              ),
              if (_isPlaying) ...[
                const SizedBox(height: 20),
                const Text(
                  "Sound is playing...",
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
