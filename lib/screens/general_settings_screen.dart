// lib/screens/general_settings_screen.dart
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class GeneralSettingsScreen extends StatefulWidget {
  const GeneralSettingsScreen({super.key});

  @override
  State<GeneralSettingsScreen> createState() => _GeneralSettingsScreenState();
}

class _GeneralSettingsScreenState extends State<GeneralSettingsScreen> {
  bool darkMode = false;
  bool haptics = true;
  bool notifications = true;
  double alarmVolume = 0.7; // 0.0 - 1.0

  static const String _kDarkMode = 'pref_dark_mode';
  static const String _kHaptics = 'pref_haptics';
  static const String _kNotifications = 'pref_notifications';
  static const String _kAlarmVolume = 'pref_alarm_volume';

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      darkMode = prefs.getBool(_kDarkMode) ?? false;
      haptics = prefs.getBool(_kHaptics) ?? true;
      notifications = prefs.getBool(_kNotifications) ?? true;
      alarmVolume = prefs.getDouble(_kAlarmVolume) ?? 0.7;
    });
  }

  Future<void> _savePref(String key, Object value) async {
    final prefs = await SharedPreferences.getInstance();
    if (value is bool) await prefs.setBool(key, value);
    if (value is double) await prefs.setDouble(key, value);
    // no snackbar every change to avoid spam — show small hint on Save/Reset
  }

  Future<void> _saveAllAndPop() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kDarkMode, darkMode);
    await prefs.setBool(_kHaptics, haptics);
    await prefs.setBool(_kNotifications, notifications);
    await prefs.setDouble(_kAlarmVolume, alarmVolume);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Saved settings')),
    );
    Navigator.pop(context);
  }

  Future<void> _resetDefaults() async {
    setState(() {
      darkMode = false;
      haptics = true;
      notifications = true;
      alarmVolume = 0.7;
    });
    await _savePref(_kDarkMode, darkMode);
    await _savePref(_kHaptics, haptics);
    await _savePref(_kNotifications, notifications);
    await _savePref(_kAlarmVolume, alarmVolume);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Reset to defaults')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('App Settings'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          TextButton(
            onPressed: _resetDefaults,
            child: const Text('Reset', style: TextStyle(color: Colors.white70)),
          )
        ],
      ),
      body: Container(
        padding: const EdgeInsets.all(18),
        decoration: const BoxDecoration(
          gradient: LinearGradient(
              colors: [Color(0xFF141328), Color(0xFF2A2140)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight),
        ),
        child: Column(
          children: [
            SwitchListTile(
              value: darkMode,
              onChanged: (v) {
                setState(() => darkMode = v);
                _savePref(_kDarkMode, v);
              },
              title: const Text('Dark mode',
                  style: TextStyle(color: Colors.white)),
              subtitle: const Text('Use dark theme even in system light mode',
                  style: TextStyle(color: Colors.white70)),
              activeColor: Colors.blueAccent,
            ),
            const Divider(color: Colors.white12),
            SwitchListTile(
              value: haptics,
              onChanged: (v) {
                setState(() => haptics = v);
                _savePref(_kHaptics, v);
              },
              title: const Text('Haptic feedback',
                  style: TextStyle(color: Colors.white)),
              subtitle: const Text('Vibrate on interactions',
                  style: TextStyle(color: Colors.white70)),
              activeColor: Colors.blueAccent,
            ),
            const Divider(color: Colors.white12),
            SwitchListTile(
              value: notifications,
              onChanged: (v) {
                setState(() => notifications = v);
                _savePref(_kNotifications, v);
              },
              title: const Text('Notifications',
                  style: TextStyle(color: Colors.white)),
              subtitle: const Text('Allow alarms & reminders',
                  style: TextStyle(color: Colors.white70)),
              activeColor: Colors.blueAccent,
            ),
            const Divider(color: Colors.white12),
            ListTile(
              title: const Text('Default alarm volume',
                  style: TextStyle(color: Colors.white)),
              subtitle: Text('${(alarmVolume * 100).round()}%',
                  style: const TextStyle(color: Colors.white70)),
            ),
            Slider(
              value: alarmVolume,
              min: 0.0,
              max: 1.0,
              divisions: 10,
              label: '${(alarmVolume * 100).round()}%',
              onChanged: (v) {
                setState(() => alarmVolume = v);
                _savePref(_kAlarmVolume, v);
              },
            ),
            const Spacer(),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _saveAllAndPop,
                    child: const Text('Save'),
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}
