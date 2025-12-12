// lib/models/alarm.dart

import 'dart:convert';
import 'dart:math';

class Alarm implements Comparable<Alarm> {
  /// Unique identifier for this alarm. If null in constructor a new id is generated.
  final int id;

  /// Exact DateTime when the alarm should fire (local time).
  final DateTime time;

  /// Human-friendly description shown in UI.
  final String description;

  /// Whether this alarm is enabled (should be scheduled).
  final bool enabled;

  /// Optional cycle index (if you compute cycle-based alarms).
  final int? cycle;

  /// Optional tone asset string (e.g. 'assets/audio/drizzling.mp3' or native raw name).
  final String? toneAsset;

  /// Repeat days represented as integers 0..6 (Sunday = 0). Empty set = no repeat.
  final Set<int> repeatDays;

  Alarm({
    int? id,
    required this.time,
    this.description = '',
    this.enabled = false,
    this.cycle,
    this.toneAsset,
    Set<int>? repeatDays,
  })  : id = id ?? Alarm.generateId(),
        repeatDays =
            repeatDays == null ? <int>{} : Set<int>.unmodifiable(repeatDays);

  /// Create a copy with modifications.
  Alarm copyWith({
    int? id,
    DateTime? time,
    String? description,
    bool? enabled,
    int? cycle,
    String? toneAsset,
    Set<int>? repeatDays,
  }) {
    return Alarm(
      id: id ?? this.id,
      time: time ?? this.time,
      description: description ?? this.description,
      enabled: enabled ?? this.enabled,
      cycle: cycle ?? this.cycle,
      toneAsset: toneAsset ?? this.toneAsset,
      repeatDays: repeatDays ?? this.repeatDays,
    );
  }

  /// Convert to a Map suitable for persistence (SharedPreferences / Json).
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'time_iso': time.toIso8601String(),
      'description': description,
      'enabled': enabled,
      'cycle': cycle,
      'toneAsset': toneAsset,
      'repeatDays': repeatDays.toList(),
    };
  }

  factory Alarm.fromMap(Map<String, dynamic> map) {
    return Alarm(
      id: map['id'] is int
          ? map['id'] as int
          : int.tryParse(map['id']?.toString() ?? ''),
      time: DateTime.parse(map['time_iso'] as String),
      description: map['description']?.toString() ?? '',
      enabled: map['enabled'] == true,
      cycle: map['cycle'] is int
          ? map['cycle'] as int
          : (map['cycle'] == null
              ? null
              : int.tryParse(map['cycle'].toString())),
      toneAsset: map['toneAsset']?.toString(),
      repeatDays:
          (map['repeatDays'] as List<dynamic>?)?.map((e) => e as int).toSet() ??
              <int>{},
    );
  }

  String toJson() => jsonEncode(toMap());

  factory Alarm.fromJson(String source) =>
      Alarm.fromMap(jsonDecode(source) as Map<String, dynamic>);

  /// Generate a high-entropy unique id. Uses microsecond timestamp XOR a small random
  /// value. Collisions are exceedingly unlikely for normal UI flows.
  static int generateId() {
    final ts = DateTime.now().microsecondsSinceEpoch;
    final rnd = Random().nextInt(1 << 20); // up to ~1M
    return ts ^ rnd;
  }

  /// Format time to a 'h:mm AM/PM' string.
  String formatTo12h() {
    final hour = (time.hour % 12) == 0 ? 12 : (time.hour % 12);
    final minute = time.minute.toString().padLeft(2, '0');
    final ap = time.hour >= 12 ? 'PM' : 'AM';
    return '\$hour:\$minute \$ap';
  }

  @override
  int compareTo(Alarm other) => time.compareTo(other.time);

  @override
  bool operator ==(Object other) =>
      identical(this, other) || (other is Alarm && other.id == id);

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() =>
      'Alarm(id: \$id, time: \$time, enabled: \$enabled, desc: "\$description")';
}
