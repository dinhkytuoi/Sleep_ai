// lib/services/sound_manager.dart
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class SoundManager {
  static const String _key = 'downloaded_sounds';

  // Stored as Map<assetPath, localPath>
  static Future<Map<String, String>> _readMap() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonStr = prefs.getString(_key);
    if (jsonStr == null) return {};
    final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
    return decoded.map((k, v) => MapEntry(k, v as String));
  }

  static Future<void> _writeMap(Map<String, String> map) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(map));
  }

  static Future<bool> isDownloaded(String assetPath) async {
    final map = await _readMap();
    return map.containsKey(assetPath);
  }

  static Future<String?> getLocalPath(String assetPath) async {
    final map = await _readMap();
    return map[assetPath];
  }

  static Future<void> addDownloaded(String assetPath, String localPath) async {
    final map = await _readMap();
    map[assetPath] = localPath;
    await _writeMap(map);
  }

  static Future<void> removeDownloaded(String assetPath) async {
    final map = await _readMap();
    map.remove(assetPath);
    await _writeMap(map);
  }

  static Future<Map<String, String>> getAllDownloaded() async {
    return await _readMap();
  }
}
