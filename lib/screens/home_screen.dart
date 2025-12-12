import 'package:path/path.dart' as p;
import 'dart:async';
import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:file_selector/file_selector.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/server_api.dart';
import '../services/alarm_service.dart';
import 'alarm_screen.dart';
import 'alarm_ring_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  TimeOfDay bedTime = const TimeOfDay(hour: 22, minute: 0);
  TimeOfDay wakeTime = const TimeOfDay(hour: 6, minute: 30);

  List<bool> toggles = [true, false, true];
  List<Map<String, dynamic>> _generatedAlarms = [];

  final _server = ServerApi('http://10.0.2.2:8000');
  int currentIndex = 0;
  bool _calculating = false;
  bool _uploading = false;
  String? _selectedFileName;
  String? _selectedFilePath;

  Timer? _checkTimer;
  final Set<String> _triggeredAlarms =
      {}; // 🔔 Lưu báo thức đã kêu để tránh lặp

  @override
  void initState() {
    super.initState();
    _init(); // ✅ thay vì gọi 2 hàm rời
  }

  Future<void> _init() async {
    await _loadAlarmPrefs(); // đợi load xong
    _checkAlarmsAndNavigate(); // check ngay 1 lần để không lỡ nhịp
    _startCheckTimer(); // rồi mới chạy timer
  }

  @override
  void dispose() {
    _checkTimer?.cancel();
    super.dispose();
  }

  // ⏱ Kiểm tra thời gian mỗi 30 giây
  void _startCheckTimer() {
    _checkTimer?.cancel();
    _checkTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      _checkAlarmsAndNavigate();
    });
  }

  Future<void> _checkAlarmsAndNavigate() async {
    if (_generatedAlarms.isEmpty) return;

    final now = DateTime.now();

    for (int i = 0; i < _generatedAlarms.length; i++) {
      final alarm = _generatedAlarms[i];
      final isEnabled = i < toggles.length ? toggles[i] : false;
      final alarmTimeStr = alarm['time']?.toString() ?? '';
      if (!isEnabled || alarmTimeStr.isEmpty) continue;
      if (_triggeredAlarms.contains(alarmTimeStr)) continue;

      final alarmDt = _parseTimeString(alarmTimeStr, now);
      if (alarmDt == null) continue;

      final diffSec = alarmDt.difference(now).inSeconds;

      // ✅ Cho phép lệch nhẹ ±2s để không lỡ tick
      if (diffSec.abs() <= 2) {
        _triggeredAlarms.add(alarmTimeStr);
        await _navigateToAlarmRing(alarmTimeStr);
      }
    }
  }

  Future<void> _navigateToAlarmRing(String time) async {
    if (!mounted) return;

    // ✅ Lấy tone đã chọn (chính là key 'chosen_tone' bạn đang dùng bên AlarmScreen)
    final prefs = await SharedPreferences.getInstance();
    final tone = prefs.getString('chosen_tone') ?? 'assets/audio/drizzling.mp3';

    // ❌ Bỏ const vì sẽ truyền tham số động
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => AlarmRingScreen(toneAsset: tone),
      ),
    );

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('⏰ Alarm ringing at $time')),
    );
  }

  /// Helper parse "HH:mm" hoặc "h:mm AM/PM"
  DateTime? _parseTimeString(String s, DateTime reference,
      {bool allowNextDay = true}) {
    s = s.trim();
    final hhmm24 = RegExp(r'^(\d{1,2}):(\d{2})$');
    final ampm = RegExp(r'^(\d{1,2}):(\d{2})\s*([AaPp][Mm])$');

    Match? m;
    int hour = 0, min = 0;
    if ((m = hhmm24.firstMatch(s)) != null) {
      hour = int.parse(m!.group(1)!);
      min = int.parse(m.group(2)!);
    } else if ((m = ampm.firstMatch(s)) != null) {
      hour = int.parse(m!.group(1)!);
      min = int.parse(m.group(2)!);
      final ap = m.group(3)!.toLowerCase();
      if (ap == 'pm' && hour != 12) hour += 12;
      if (ap == 'am' && hour == 12) hour = 0;
    } else {
      return null;
    }

    DateTime dt =
        DateTime(reference.year, reference.month, reference.day, hour, min);
    if (allowNextDay && dt.isBefore(reference)) {
      dt = dt.add(const Duration(days: 1));
    }
    return dt;
  }

  /// 🧠 Lưu trạng thái báo thức (toggles và times)
  Future<void> _saveAlarmPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final toggleStrings = toggles.map((v) => v.toString()).toList();
    final timeStrings =
        _generatedAlarms.map((a) => a['time'].toString()).toList();

    await prefs.setStringList('alarm_toggles', toggleStrings);
    await prefs.setStringList('alarm_times', timeStrings);

    print('💾 [SAVE] Saved alarm_toggles=$toggleStrings');
    print('💾 [SAVE] Saved alarm_times=$timeStrings');
  }

  /// 🧠 Load lại khi mở app
  Future<void> _loadAlarmPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final toggleStrings = prefs.getStringList('alarm_toggles') ?? [];
    final timeStrings = prefs.getStringList('alarm_times') ?? [];

    setState(() {
      toggles = toggleStrings.map((s) => s == 'true').toList();
      _generatedAlarms = timeStrings
          .map((t) => {'time': t, 'description': 'Loaded from prefs'})
          .toList();
    });

    print('📂 [LOAD] alarm_toggles=$toggleStrings');
    print('📂 [LOAD] alarm_times=$timeStrings');
    print('📂 Loaded alarms: $_generatedAlarms');
  }

  String _formatTimeOfDay(TimeOfDay t) {
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    return '$hh:$mm';
  }

  Future<void> pickTime(bool isBed) async {
    final initial = isBed ? bedTime : wakeTime;
    final result = await showTimePicker(context: context, initialTime: initial);
    if (result != null) {
      setState(() {
        if (isBed) {
          bedTime = result;
        } else {
          wakeTime = result;
        }
      });
    }
  }

  Future<void> _pickDatasetFile() async {
    try {
      final XTypeGroup edfGroup = XTypeGroup(
        label: 'edf',
        extensions: ['edf', 'EDF', 'bin'],
      );
      final XFile? file = await openFile(acceptedTypeGroups: [edfGroup]);
      if (file == null) return;

      print('[pick] picked.path=${file.path}, picked.name=${file.name}');
      String displayName = file.name;

      if (!displayName.toLowerCase().endsWith('.edf')) {
        displayName = p.setExtension(displayName, '.edf');
      }

      setState(() {
        _selectedFileName = displayName;
        _selectedFilePath = file.path;
      });
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Pick failed: $e')));
    }
  }

  Future<void> _uploadSelectedFile() async {
    if (_selectedFilePath == null) return;
    setState(() => _uploading = true);
    try {
      final f = File(_selectedFilePath!);
      final resp = await _server.uploadFile(
        f,
        title: _selectedFileName,
        forceEdf: true,
      );

      final serverFilename =
          resp['filename'] ?? resp['file'] ?? _selectedFileName;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload successful: $serverFilename')),
      );

      setState(() {
        _selectedFileName = serverFilename?.toString();
      });
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Upload error: $e')));
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  /// Format DateTime to "h:mm AM/PM"
  String _formatTo12h(DateTime dt) {
    final hour = dt.hour % 12 == 0 ? 12 : dt.hour % 12;
    final minute = dt.minute.toString().padLeft(2, '0');
    final ap = dt.hour >= 12 ? 'PM' : 'AM';
    return '$hour:$minute $ap';
  }

  /// Check if dt is within [start, end). Handles overnight when end <= start by adding 1 day to end.
  bool _isWithin(DateTime dt, DateTime start, DateTime end) {
    if (!end.isAfter(start)) {
      end = end.add(const Duration(days: 1));
      if (dt.isBefore(start)) dt = dt.add(const Duration(days: 1));
    }
    return !dt.isBefore(start) && dt.isBefore(end);
  }

  Future<void> _calculateAlarms() async {
    if (_selectedFileName == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Please pick & upload an EDF file first.')),
      );
      return;
    }

    setState(() => _calculating = true);
    try {
      final bed = _formatTimeOfDay(bedTime);
      final wake = _formatTimeOfDay(wakeTime);

      // References
      final today = DateTime.now();
      DateTime bedRefNoBuffer = DateTime(
          today.year, today.month, today.day, bedTime.hour, bedTime.minute);
      DateTime wakeRef = DateTime(
          today.year, today.month, today.day, wakeTime.hour, wakeTime.minute);

      // ensure wake is after bed (may be next day)
      if (!wakeRef.isAfter(bedRefNoBuffer))
        wakeRef = wakeRef.add(const Duration(days: 1));

      // Buffer used for cycle-based slots (same idea as server heuristic)
      final bedRefWithBuffer = bedRefNoBuffer.add(const Duration(minutes: 15));

      // If not enough time for 1 cycle (90 min) starting from buffered bed -> show warning and stop
      final availableMinutes = wakeRef.difference(bedRefWithBuffer).inMinutes;
      if (availableMinutes < 90) {
        // clear any previous alarms
        setState(() => _generatedAlarms = []);
        // show dialog / snackbar
        showDialog(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text('Không đủ thời gian'),
            content: Text(
                'Khoảng thời gian từ lúc ngủ (+15 min) đến lúc thức chỉ có ${availableMinutes} phút — không đủ 1 chu kỳ 90 phút.'),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('OK'))
            ],
          ),
        );
        return;
      }

      // Call server (we still can use mode 1 to get model suggestions)
      final res = await _server.calculate(
        filename: _selectedFileName!,
        bedTime: bed,
        wakeTime: wake,
        age: "25",
        gender: "nam",
        mode: "1",
      );

      final rawAlarms = (res['alarms'] as List<dynamic>?)
              ?.map((e) => Map<String, dynamic>.from(e))
              .toList() ??
          [];

      // Parse model alarms (use bedRefNoBuffer as reference so we don't wrongly exclude model suggestions)
      final List<Map<String, dynamic>> parsedModel = [];
      for (final a in rawAlarms) {
        final raw = (a['time'] ?? a['Time'] ?? a.toString()).toString().trim();
        final dt = _parseTimeString(raw, bedRefNoBuffer, allowNextDay: true);
        if (dt == null) continue;
        if (_isWithin(dt, bedRefNoBuffer, wakeRef)) {
          parsedModel.add({
            'time_dt': dt,
            'time_str': _formatTo12h(dt),
            'description': a['description'] ?? a['desc'] ?? '',
          });
        }
      }

      // Determine cycle slots by generating until slot >= wakeRef
      final List<DateTime> cycleSlots = [];
      DateTime t = bedRefWithBuffer;
      while (t.isBefore(wakeRef)) {
        cycleSlots.add(t);
        t = t.add(const Duration(minutes: 90));
      }

      // slotWindow: how far from the slot we accept model suggestions.
      final slotWindow = const Duration(minutes: 10); // adjust to taste

      // Combine model / cycle into candidate alarms, keep time_dt for sorting
      final List<Map<String, dynamic>> combined = [];
      for (var i = 0; i < cycleSlots.length; i++) {
        final slot = cycleSlots[i];
        Map<String, dynamic>? best;
        int bestDiff = 999999;

        for (final m in parsedModel) {
          final dt = m['time_dt'] as DateTime;
          final diff = dt.difference(slot).inMinutes.abs();
          if (dt.isAfter(slot.subtract(slotWindow)) &&
              dt.isBefore(slot.add(slotWindow))) {
            if (diff < bestDiff) {
              bestDiff = diff;
              best = m;
            }
          }
        }

        if (best != null) {
          combined.add({
            'time_dt': best['time_dt'],
            'time_str': best['time_str'],
            'description': best['description'],
            'cycle_index': i + 1,
          });
        } else {
          // fallback: use exact slot time as cycle-based
          combined.add({
            'time_dt': slot,
            'time_str': _formatTo12h(slot),
            'description': 'Cycle-based',
            'cycle_index': i + 1,
          });
        }
      }

      // Sort combined by chronological order, dedupe very-close alarms
      combined.sort((a, b) =>
          (a['time_dt'] as DateTime).compareTo(b['time_dt'] as DateTime));

      final List<Map<String, dynamic>> finalAlarmsWithDt = [];
      DateTime? last;
      for (final p in combined) {
        final dt = p['time_dt'] as DateTime;
        if (last != null && dt.difference(last).inMinutes.abs() < 2) continue;

        final desc = (p['description'] ?? '').toString();
        final cycleIdx = p['cycle_index'] ?? (finalAlarmsWithDt.length + 1);
        final friendlyDesc = 'Cycle $cycleIdx — $desc';

        finalAlarmsWithDt.add({
          'time_dt': dt,
          'time': p['time_str'],
          'description': friendlyDesc,
          'cycle': cycleIdx,
        });

        last = dt;
        if (finalAlarmsWithDt.length >= 6) break;
      }

      // --- NEW: reorder by proximity to wakeRef (closest first) ---
      // This moves the alarm whose DateTime is nearest to wakeRef to the top.
      finalAlarmsWithDt.sort((a, b) {
        final aDt = a['time_dt'] as DateTime;
        final bDt = b['time_dt'] as DateTime;
        final aDist = aDt.difference(wakeRef).inSeconds.abs();
        final bDist = bDt.difference(wakeRef).inSeconds.abs();
        return aDist.compareTo(bDist);
      });

      // Optionally, keep chronological order within same-distance group.
      // (If you prefer stable cycle ordering after placing nearest-first, you can tweak here.)

      // Trim to max 6 (already limited, but ensure)
      final finalAlarms = finalAlarmsWithDt.take(6).toList();

      setState(() {
        _generatedAlarms = finalAlarms.map((a) {
          // keep 'time_dt' for internal use (scheduling, sorting), but UI uses 'time'
          return {
            'time_dt': a['time_dt'],
            'time': a['time'],
            'description': a['description'],
            'cycle': a['cycle'],
          };
        }).toList();

        // Ensure toggles list is long enough and default enabled = true
        if (toggles.length < _generatedAlarms.length) {
          toggles = [
            ...toggles,
            ...List.filled(_generatedAlarms.length - toggles.length, true)
          ];
        }
      });

      // Lưu vào SharedPreferences (saves only time strings)
      await _saveAlarmPrefs();

      final alarmTexts = _generatedAlarms
          .map((a) => '${a['time']} — ${a['description']}')
          .join('\n');

      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Calculated Alarms'),
          content:
              Text(alarmTexts.isNotEmpty ? alarmTexts : 'No alarms returned'),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'))
          ],
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Calculate error: $e')));
    } finally {
      if (mounted) setState(() => _calculating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final today = DateFormat("dd MMM, yyyy").format(DateTime.now());

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/background.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "Welcome to Sleepora",
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          today,
                          style: const TextStyle(
                              color: Colors.white70, fontSize: 15),
                        ),
                      ],
                    ),
                    GestureDetector(
                      onTap: () => Navigator.pushNamed(context, '/profile'),
                      child: const CircleAvatar(
                        radius: 20,
                        backgroundImage: AssetImage("assets/images/avatar.png"),
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 20),

                // Upload box
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color.fromARGB(51, 255, 255, 255),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Expanded(
                        child: _selectedFileName == null
                            ? const Text(
                                'No dataset selected.\nTap Pick to choose an EDF file.',
                                style: TextStyle(color: Colors.white70),
                              )
                            : Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Text(
                                    'Selected dataset:',
                                    style: TextStyle(
                                        color: Colors.white70, fontSize: 12),
                                  ),
                                  const SizedBox(height: 6),
                                  Text(
                                    _selectedFileName!,
                                    style: const TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold),
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ],
                              ),
                      ),
                      const SizedBox(width: 8),
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton.icon(
                            onPressed: _pickDatasetFile,
                            icon: const Icon(Icons.upload_file),
                            label: const Text('Pick'),
                          ),
                          const SizedBox(height: 8),
                          _uploading
                              ? const SizedBox(
                                  width: 110,
                                  height: 34,
                                  child: Center(
                                      child: CircularProgressIndicator()))
                              : ElevatedButton.icon(
                                  onPressed: _selectedFilePath == null
                                      ? null
                                      : _uploadSelectedFile,
                                  icon: const Icon(Icons.cloud_upload),
                                  label: const Text('Upload'),
                                  style: ElevatedButton.styleFrom(
                                      backgroundColor: Colors.green),
                                ),
                        ],
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 30),

                // Time pickers
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildTimePicker("In-bed", bedTime, () => pickTime(true)),
                    _buildTimePicker(
                        "Wake-up", wakeTime, () => pickTime(false)),
                  ],
                ),

                const SizedBox(height: 35),

                // Calculate button
                Center(
                  child: _calculating
                      ? const CircularProgressIndicator()
                      : ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF4284F5),
                            padding: const EdgeInsets.symmetric(
                                horizontal: 28, vertical: 12),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          onPressed: _calculateAlarms,
                          child: const Text(
                            "Calculate",
                            style: TextStyle(fontSize: 18, color: Colors.white),
                          ),
                        ),
                ),

                const SizedBox(height: 30),

                Expanded(
                  child: ListView(
                    physics: const BouncingScrollPhysics(),
                    children: [
                      if (_generatedAlarms.isEmpty) ...[
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 40),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: const [
                              Icon(Icons.hourglass_empty,
                                  size: 56, color: Colors.white24),
                              SizedBox(height: 12),
                              Padding(
                                padding: EdgeInsets.symmetric(horizontal: 20.0),
                                child: Text(
                                  'Click "Calculate".',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                      color: Colors.white70, fontSize: 15),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ] else
                        ..._generatedAlarms.asMap().entries.map((entry) {
                          final idx = entry.key;
                          final a = entry.value;
                          final time = a['time']?.toString() ??
                              a['Time']?.toString() ??
                              'Unknown';
                          final desc = a['description']?.toString() ?? '';
                          return buildAlarm(time, desc, idx);
                        }),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      bottomNavigationBar: _buildBottomNavBar(),
    );
  }

  Widget _buildTimePicker(String label, TimeOfDay time, VoidCallback onTap) {
    return Column(
      children: [
        Text(label,
            style: const TextStyle(color: Colors.white70, fontSize: 18)),
        const SizedBox(height: 5),
        GestureDetector(
          onTap: onTap,
          child: Row(
            children: [
              Text(
                time.format(context),
                style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.white),
              ),
              const SizedBox(width: 6),
              const Icon(Icons.edit, size: 18, color: Colors.white70),
            ],
          ),
        ),
      ],
    );
  }

  Widget buildAlarm(String time, String desc, int index) {
    final toggle = index < toggles.length ? toggles[index] : false;

    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => AlarmScreen(
              alarmTime: time,
              enabled: toggle,
              onToggle: (v) async {
                if (index < toggles.length) {
                  setState(() => toggles[index] = v);
                } else {
                  setState(() => toggles.add(v));
                }

                // Lưu trạng thái mới
                await _saveAlarmPrefs();

                // 🔔 Đặt hoặc hủy báo thức
                if (v) {
                  final alarmTime = time;
                  final dt = _parseTimeString(alarmTime, DateTime.now());
                  if (dt != null) {
                    final prefs = await SharedPreferences.getInstance();
                    final selectedSound = prefs.getString('chosen_tone') ??
                        'assets/audio/drizzling.mp3';

                    String? soundRawName;
                    if (selectedSound.startsWith('audio/') ||
                        selectedSound.startsWith('assets/audio/')) {
                      soundRawName = null;
                    } else {
                      soundRawName = selectedSound.replaceAll('.mp3', '');
                    }

                    await AlarmService.scheduleAlarm(
                      id: index + 1,
                      dateTimeLocal: dt,
                      payload: selectedSound,
                      soundRawName: soundRawName ?? 'drizzling',
                    );

                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('⏰ Alarm set for $alarmTime')),
                    );
                  }
                } else {
                  await AlarmService.cancel(index + 1);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Alarm #${index + 1} canceled')),
                  );
                }
              },
            ),
          ),
        ).then((_) {
          // Cập nhật lại trạng thái khi quay về từ AlarmScreen
          if (mounted) setState(() {});
        });
      },
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Flexible(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    time,
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: toggle ? Colors.greenAccent : Colors.white,
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(desc,
                      style: const TextStyle(color: Colors.white70),
                      overflow: TextOverflow.ellipsis),
                ],
              ),
            ),
            Switch(
              value: toggle,
              onChanged: (v) async {
                if (index < toggles.length) {
                  setState(() => toggles[index] = v);
                } else {
                  setState(() => toggles.add(v));
                }

                // Lưu trạng thái mới
                await _saveAlarmPrefs();

                // 🔔 Đặt hoặc hủy báo thức
                if (v) {
                  final alarmTime = time;
                  final dt = _parseTimeString(alarmTime, DateTime.now());
                  if (dt != null) {
                    final prefs = await SharedPreferences.getInstance();
                    final selectedSound = prefs.getString('selectedSound') ??
                        'assets/audio/drizzling.mp3';

                    String? soundRawName;
                    if (selectedSound.startsWith('audio/') ||
                        selectedSound.startsWith('assets/audio/')) {
                      soundRawName = null;
                    } else {
                      soundRawName = selectedSound.replaceAll('.mp3', '');
                    }

                    await AlarmService.scheduleAlarm(
                      id: index + 1,
                      dateTimeLocal: dt,
                      payload: selectedSound,
                      soundRawName: soundRawName ?? 'drizzling',
                    );

                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('⏰ Alarm set for $alarmTime')),
                    );
                  }
                } else {
                  await AlarmService.cancel(index + 1);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Alarm #${index + 1} canceled')),
                  );
                }
              },
              activeThumbColor: Colors.greenAccent,
              activeTrackColor: const Color.fromARGB(127, 105, 240, 174),
              inactiveThumbColor: Colors.grey,
              inactiveTrackColor: Colors.grey.shade700,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomNavBar() {
    return Container(
      decoration: BoxDecoration(
        color: const Color.fromARGB(230, 28, 16, 77),
        boxShadow: [
          BoxShadow(
            color: const Color.fromARGB(41, 55, 15, 156),
            offset: const Offset(0, -1),
            blurRadius: 4,
          ),
        ],
      ),
      child: ClipRRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 2, sigmaY: 2),
          child: BottomNavigationBar(
            currentIndex: currentIndex,
            onTap: (i) {
              if (i == 0) {
                setState(() => currentIndex = 0);
              } else if (i == 1) {
                Navigator.pushReplacementNamed(context, '/sound');
              } else if (i == 2) {
                Navigator.pushReplacementNamed(context, '/settings');
              }
            },
            backgroundColor: Colors.transparent,
            elevation: 0,
            selectedItemColor: Colors.white,
            unselectedItemColor: Colors.white70,
            items: const [
              BottomNavigationBarItem(
                  icon: Icon(Icons.alarm_rounded), label: "Alarm"),
              BottomNavigationBarItem(
                  icon: Icon(Icons.bar_chart_rounded), label: "Sound"),
              BottomNavigationBarItem(
                  icon: Icon(Icons.settings_rounded), label: "Settings"),
            ],
          ),
        ),
      ),
    );
  }
}
