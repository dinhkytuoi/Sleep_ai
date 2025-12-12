// lib/services/alarm_manager.dart
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/alarm.dart';
import 'alarm_service.dart';

class AlarmManager {
  static const _prefsKey = 'alarms_v1';

  /// Load list of alarms from prefs
  static Future<List<Alarm>> loadAlarms() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getStringList(_prefsKey) ?? [];
    final out = raw
        .map((s) {
          try {
            return Alarm.fromJson(s);
          } catch (_) {
            return null;
          }
        })
        .whereType<Alarm>()
        .toList();
    out.sort();
    return out;
  }

  /// Save list of alarms to prefs
  static Future<void> saveAlarms(List<Alarm> alarms) async {
    final prefs = await SharedPreferences.getInstance();
    final serialized = alarms.map((a) => a.toJson()).toList();
    await prefs.setStringList(_prefsKey, serialized);
  }

  /// Schedule all enabled alarms (useful on app startup)
  static Future<void> scheduleEnabledAlarms(List<Alarm> alarms) async {
    for (final a in alarms) {
      if (a.enabled) {
        await AlarmService.scheduleAlarm(
          id: a.id,
          dateTimeLocal: a.time,
          // title: 'Sleepora Alarm',
          // body: a.description,
          // repeatDaily: false,
          soundRawName: _rawNameFromAsset(a.toneAsset ?? 'drizzling'),
          payload: 'alarm:${a.id}',
        );
      }
    }
  }

  /// Add or update an alarm: persist and schedule/cancel accordingly
  static Future<void> upsertAlarm(Alarm alarm) async {
    final list = await loadAlarms();
    final idx = list.indexWhere((x) => x.id == alarm.id);
    if (idx >= 0) {
      list[idx] = alarm;
    } else {
      list.add(alarm);
    }
    await saveAlarms(list);

    if (alarm.enabled) {
      await AlarmService.scheduleAlarm(
        id: alarm.id,
        dateTimeLocal: alarm.time,
        // title: 'Sleepora Alarm',
        // body: alarm.description,
        // repeatDaily: false,
        soundRawName: _rawNameFromAsset(alarm.toneAsset ?? 'drizzling'),
        payload: 'alarm:${alarm.id}',
      );
    } else {
      await AlarmService.cancel(alarm.id);
    }
  }

  static Future<void> deleteAlarm(int id) async {
    final list = await loadAlarms();
    list.removeWhere((x) => x.id == id);
    await saveAlarms(list);
    await AlarmService.cancel(id);
  }

  static String _rawNameFromAsset(String asset) {
    // If asset is e.g. 'assets/audio/drizzling.mp3' -> return 'drizzling'
    try {
      final name = asset.split('/').last.split('.').first;
      return name;
    } catch (_) {
      return asset;
    }
  }
}
