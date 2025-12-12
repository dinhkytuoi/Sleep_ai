// lib/screens/alarm_screen.dart
import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/sound_manager.dart';
import '../services/alarm_service.dart';
import 'alarm_ring_screen.dart';

class AlarmScreen extends StatefulWidget {
  final String alarmTime;
  final bool enabled;
  final ValueChanged<bool>? onToggle;

  const AlarmScreen({
    super.key,
    required this.alarmTime,
    this.enabled = false,
    this.onToggle,
  });

  @override
  State<AlarmScreen> createState() => _AlarmScreenState();
}

class _AlarmScreenState extends State<AlarmScreen> {
  late bool enabledLocal;
  bool vibration = true;
  bool snooze = true;
  final Set<int> selectedDays = {};
  final AudioPlayer _previewPlayer = AudioPlayer();
  bool _isPreviewPlaying = false;
  String? _chosenToneAsset;
  Timer? _checkTimer;
  bool _alarmTriggered = false; // ✅ Đảm bảo chỉ báo thức 1 lần

  // Lấy tên hiển thị của tone đang chọn
  String _selectedToneTitle() {
    final asset = _chosenToneAsset ?? 'assets/audio/drizzling.mp3';
    final found = availableTones.where((t) => t['asset'] == asset);
    if (found.isNotEmpty) return found.first['title']!;
    // Nếu không nằm trong danh sách mặc định thì coi như "Custom"
    final fileName = asset.split('/').last;
    return 'Custom ($fileName)';
  }

// Nghe thử / dừng nghe thử nhạc chuông hiện tại
  Future<void> _previewSelectedTone() async {
    final asset = _chosenToneAsset ?? 'assets/audio/drizzling.mp3';

    // Toggle: nếu đang phát thì dừng
    if (_isPreviewPlaying) {
      await _previewPlayer.stop();
      if (mounted) setState(() => _isPreviewPlaying = false);
      return;
    }

    // Tìm đường dẫn local đã tải (nếu có)
    final downloadedMap = await SoundManager.getAllDownloaded();
    String? candidate = downloadedMap[asset] ??
        downloadedMap[asset.replaceFirst('assets/', '')];

    try {
      if (candidate != null &&
          candidate.isNotEmpty &&
          File(candidate).existsSync()) {
        await _previewPlayer.play(DeviceFileSource(candidate));
      } else {
        // audioplayers AssetSource cần path tương đối (không có 'assets/')
        final relative = asset.startsWith('assets/')
            ? asset.substring('assets/'.length)
            : asset;
        await _previewPlayer.play(AssetSource(relative));
      }
      if (mounted) setState(() => _isPreviewPlaying = true);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Không phát được âm thanh: $e')),
        );
      }
    }
  }

  final List<Map<String, String>> availableTones = [
    {'title': 'Drizzling', 'asset': 'assets/audio/drizzling.mp3'},
    {'title': 'Raindrops Drum', 'asset': 'assets/audio/raindrops_drum.mp3'},
    {'title': 'Summer Rain', 'asset': 'assets/audio/summer_rain.mp3'},
    {'title': 'Stepping Rain', 'asset': 'assets/audio/stepping_rain.mp3'},
    {'title': 'Rain in Forest', 'asset': 'assets/audio/rain_forest.mp3'},
    {'title': 'Showers on Window', 'asset': 'assets/audio/showers_window.mp3'},
    {'title': 'Gentle Stream', 'asset': 'assets/audio/gentle_stream.mp3'},
    {'title': 'Night Thunder', 'asset': 'assets/audio/night_thunder.mp3'},
  ];

  @override
  void initState() {
    super.initState();
    enabledLocal = widget.enabled;
    _loadChosenTone();

    _previewPlayer.onPlayerComplete.listen((_) {
      setState(() => _isPreviewPlaying = false);
    });

    // ✅ Kiểm tra thời gian khi bật báo thức
    if (enabledLocal) _startInTime();
  }

  @override
  void dispose() {
    _previewPlayer.dispose();
    _checkTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadChosenTone() async {
    final prefs = await SharedPreferences.getInstance();
    final tone = prefs.getString('chosen_tone');
    if (mounted) setState(() => _chosenToneAsset = tone);
  }

  Future<void> _saveChosenTone(String asset) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('chosen_tone', asset);
    if (mounted) setState(() => _chosenToneAsset = asset);
  }

  String get nextAlarmText =>
      "Next alarm will ring at ${widget.alarmTime.toLowerCase()}";

  /// ✅ Khi toggle báo thức bật/tắt
  void _onSwitchChanged(bool v) async {
    setState(() {
      enabledLocal = v;
      _alarmTriggered = false;
    });
    widget.onToggle?.call(v);

    if (v) {
      await _setAlarmTime();
      _startInTime(); // Bắt đầu đếm thời gian thực
    } else {
      _checkTimer?.cancel();
      await AlarmService.cancel(1);
    }
  }

  /// ✅ Hàm kiểm tra thời gian thật, đúng giờ thì mở báo thức
  void _startInTime() {
    _checkTimer?.cancel();

    _checkTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!enabledLocal || _alarmTriggered) return;

      final now = DateTime.now();
      final alarmParts = widget.alarmTime.split(RegExp(r'[: ]'));
      if (alarmParts.length < 2) return;

      int hour = int.parse(alarmParts[0]);
      int minute = int.parse(alarmParts[1]);
      final ampm = widget.alarmTime.toLowerCase().contains('pm') ? 'pm' : 'am';
      if (ampm == 'pm' && hour != 12) hour += 12;
      if (ampm == 'am' && hour == 12) hour = 0;

      final current =
          DateTime(now.year, now.month, now.day, now.hour, now.minute);
      final alarm = DateTime(now.year, now.month, now.day, hour, minute);

      if (current.isAtSameMomentAs(alarm)) {
        _alarmTriggered = true;
        _triggerAlarm();
      }
    });
  }

  Future<void> _triggerAlarm() async {
    if (!mounted) return;
    await _previewPlayer.stop();

    if (mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => AlarmRingScreen(
            toneAsset: _chosenToneAsset ?? 'assets/audio/drizzling.mp3',
          ),
        ),
      );
    }
  }

  Future<void> _openTonePicker() async {
    final downloadedMap = await SoundManager.getAllDownloaded();

    final downloadedTones = availableTones.where((t) {
      final asset = t['asset']!;
      final fileName = asset.split('/').last;
      final bool exact = downloadedMap.containsKey(asset);
      final bool rel =
          downloadedMap.containsKey(asset.replaceFirst('assets/', ''));
      final bool byFileName = downloadedMap.values
          .any((local) => local.split('/').last == fileName);
      return exact || rel || byFileName;
    }).toList();

    if (downloadedTones.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No downloaded sounds found')),
      );
      return;
    }

    await showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) {
        return DraggableScrollableSheet(
          initialChildSize: 0.6,
          minChildSize: 0.4,
          maxChildSize: 0.95,
          expand: false,
          builder: (_, controller) {
            return Container(
              decoration: const BoxDecoration(
                color: Color.fromRGBO(18, 18, 23, 0.95),
                borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
              ),
              child: Column(
                children: [
                  const SizedBox(height: 12),
                  Container(
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.white24,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 18.0),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        'Select Tone',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Expanded(
                    child: ListView.builder(
                      controller: controller,
                      itemCount: downloadedTones.length,
                      itemBuilder: (context, i) {
                        final t = downloadedTones[i];
                        final asset = t['asset']!;
                        final title = t['title']!;
                        final localPath = downloadedMap[asset] ?? '';
                        final isSelected = _chosenToneAsset == asset;

                        return ListTile(
                          contentPadding: const EdgeInsets.symmetric(
                              horizontal: 18, vertical: 6),
                          leading: const Icon(Icons.music_note,
                              color: Colors.white70),
                          title: Text(title,
                              style: const TextStyle(color: Colors.white)),
                          trailing: IconButton(
                            icon: Icon(
                              isSelected
                                  ? Icons.check_circle
                                  : Icons.radio_button_unchecked,
                              color: isSelected
                                  ? Colors.greenAccent
                                  : Colors.white70,
                            ),
                            onPressed: () async {
                              await _saveChosenTone(asset);
                              await _previewPlayer.stop();
                              if (mounted) {
                                setState(() => _isPreviewPlaying = false);
                                Navigator.pop(context);
                              }
                            },
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );

    await _previewPlayer.stop();
    if (mounted) setState(() => _isPreviewPlaying = false);
  }

  @override
  Widget build(BuildContext context) {
    final parts = widget.alarmTime.split(' ');
    final hourPart = parts.isNotEmpty ? parts[0] : widget.alarmTime;
    final ampmPart = parts.length > 1 ? parts[1] : '';

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/alarmbg.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    IconButton(
                      icon:
                          const Icon(Icons.close_rounded, color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const Text(
                      "Alarm",
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 40),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: Row(
                  children: [
                    const Icon(Icons.notifications_none_rounded,
                        color: Colors.white70),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        nextAlarmText,
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 14),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 12),
              Expanded(
                child: Container(
                  margin:
                      const EdgeInsets.symmetric(horizontal: 16.0, vertical: 6),
                  padding: const EdgeInsets.only(top: 18, bottom: 20),
                  decoration: BoxDecoration(
                    color: const Color.fromARGB(18, 255, 255, 255),
                    borderRadius:
                        const BorderRadius.vertical(top: Radius.circular(20)),
                  ),
                  child: Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 20.0),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              hourPart,
                              style: const TextStyle(
                                fontSize: 56,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Padding(
                              padding: const EdgeInsets.only(bottom: 10.0),
                              child: Text(
                                ampmPart,
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.white70,
                                ),
                              ),
                            ),
                            const Spacer(),
                            Transform.scale(
                              scale: 1.05,
                              child: Switch.adaptive(
                                value: enabledLocal,
                                onChanged: _onSwitchChanged,
                                activeThumbColor:
                                    const Color.fromARGB(255, 53, 64, 168),
                                activeTrackColor:
                                    const Color.fromARGB(138, 175, 187, 224),
                                inactiveThumbColor: Colors.white70,
                                inactiveTrackColor:
                                    const Color.fromARGB(40, 255, 255, 255),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),

// --- Ringtone selector (chỉ hiển thị các sound đã tải) ---
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 10.0),
                        child: Material(
                          color: Colors.transparent,
                          child: ListTile(
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            leading: const Icon(Icons.library_music,
                                color: Colors.white70),
                            title: const Text(
                              'Ringtone',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w600),
                            ),
                            subtitle: Text(
                              _selectedToneTitle(),
                              style: const TextStyle(
                                  color: Colors.white70, fontSize: 13),
                            ),
                            onTap:
                                _openTonePicker, // mở bottom sheet chọn từ sound đã tải
                            trailing: IconButton(
                              tooltip: _isPreviewPlaying ? 'Stop' : 'Preview',
                              icon: Icon(
                                _isPreviewPlaying
                                    ? Icons.stop_circle
                                    : Icons.play_circle_fill,
                                color: Colors.white,
                              ),
                              onPressed:
                                  _previewSelectedTone, // nghe thử / dừng nghe
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Giữ nguyên hàm cũ để schedule qua AlarmService
  Future<void> _setAlarmTime({bool quickTestInMinutes = false}) async {
    DateTime now = DateTime.now();
    now = now.subtract(Duration(milliseconds: now.millisecond));

    String t = widget.alarmTime.trim();
    final ampmMatch = RegExp(r'(\d{1,2}):(\d{2})\s*([AaPp][Mm])').firstMatch(t);

    int hour = 0, minute = 0;
    if (ampmMatch != null) {
      hour = int.parse(ampmMatch.group(1)!);
      minute = int.parse(ampmMatch.group(2)!);
      final ap = ampmMatch.group(3)!.toLowerCase();
      if (ap == 'pm' && hour != 12) hour += 12;
      if (ap == 'am' && hour == 12) hour = 0;
    }

    DateTime candidate = DateTime(now.year, now.month, now.day, hour, minute);
    if (!candidate.isAfter(now))
      candidate = candidate.add(const Duration(days: 1));

    await AlarmService.scheduleAlarm(
      id: 1,
      dateTimeLocal: candidate,
      soundRawName: 'drizzling',
      payload: 'alarm:${candidate.toIso8601String()}',
    );
  }
}
