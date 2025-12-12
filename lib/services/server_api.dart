import 'package:path/path.dart' as p;
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

class ServerApi {
  final String baseUrl; // ví dụ: "http://192.168.56.1:8000"

  ServerApi(this.baseUrl);

  /// 🔹 Lấy device ID duy nhất cho Android
  static Future<String> getDeviceId() async {
    final prefs = await SharedPreferences.getInstance();

    // 🔸 Kiểm tra cache trước (để không tạo lại mỗi lần)
    final cached = prefs.getString('device_id');
    if (cached != null && cached.isNotEmpty) {
      print('✅ [DeviceID] Đã lưu trước đó: $cached');
      return cached;
    }

    String id = 'unknown_android';
    try {
      final deviceInfo = DeviceInfoPlugin();
      final info = await deviceInfo.androidInfo;

      // androidId là duy nhất cho mỗi thiết bị (trừ khi reset factory)
      id = info.id ?? info.fingerprint ?? 'unknown_android';

      print('📱 [DeviceID] Lấy từ thiết bị: $id');
    } catch (e) {
      print('⚠️ Không lấy được device info: $e');
    }

    // Nếu vẫn không lấy được thì tạo UUID
    if (id.startsWith('unknown')) {
      id = const Uuid().v4();
      print('🆕 [DeviceID] Tạo mới bằng UUID: $id');
    }

    await prefs.setString('device_id', id);
    print('💾 [DeviceID] Lưu vào SharedPreferences');
    return id;
  }

  /// 🔹 Gửi dữ liệu dự đoán đến server
  Future<Map<String, dynamic>> predict(Map<String, dynamic> input) async {
    final deviceId = await getDeviceId();
    input['device_id'] = deviceId;

    final resp = await http.post(
      Uri.parse('$baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(input),
    );

    if (resp.statusCode != 200) {
      throw Exception('predict failed: ${resp.body}');
    }

    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  /// 🔹 Lưu trạng thái (state)
  Future<void> saveState(String key, dynamic value) async {
    final deviceId = await getDeviceId();

    final resp = await http.post(
      Uri.parse('$baseUrl/save_state'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'key': key, 'value': value, 'device_id': deviceId}),
    );

    if (resp.statusCode != 200 && resp.statusCode != 201) {
      throw Exception('save_state failed: ${resp.statusCode} ${resp.body}');
    }
  }

  /// 🔹 Lấy trạng thái từ server
  Future<dynamic> getState(String key) async {
    final deviceId = await getDeviceId();
    final resp = await http.get(Uri.parse('$baseUrl/get_state/$deviceId/$key'));

    if (resp.statusCode != 200) {
      throw Exception('get_state failed: ${resp.body}');
    }

    final js = jsonDecode(resp.body) as Map<String, dynamic>;
    return js['value'];
  }

  /// 🔹 Lấy danh sách âm thanh
  Future<List<Map<String, dynamic>>> listSounds() async {
    final resp = await http.get(Uri.parse('$baseUrl/sounds'));

    if (resp.statusCode != 200) {
      throw Exception('list sounds failed: ${resp.body}');
    }

    final js = jsonDecode(resp.body);
    if (js is Map && js.containsKey('sounds')) {
      return List<Map<String, dynamic>>.from(js['sounds'] as List);
    }

    if (js is List) {
      return List<Map<String, dynamic>>.from(js);
    }

    return [];
  }

  /// 🔹 Tải file âm thanh từ server
  Future<Uint8List> downloadBytes(String fileUrl) async {
    final r = await http.get(Uri.parse(fileUrl));
    if (r.statusCode != 200) {
      throw Exception('Download failed: ${r.statusCode}');
    }
    return r.bodyBytes;
  }

  /// 🔹 Lưu file âm thanh xuống bộ nhớ app
  Future<String> downloadSoundToApp(String fileUrl, String filename) async {
    final bytes = await downloadBytes(fileUrl);
    final appDoc = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDoc.path}/sounds');
    if (!await dir.exists()) await dir.create(recursive: true);
    final file = File('${dir.path}/$filename');
    await file.writeAsBytes(bytes);
    return file.path;
  }

  /// 🔹 Upload âm thanh (legacy) — giờ gọi uploadFile và trả về JSON map
  Future<Map<String, dynamic>> uploadSound(File file,
      {String? title, bool forceEdf = false}) async {
    // Reuse uploadFile logic to keep one implementation
    return await uploadFile(file,
        title: title, forceEdf: forceEdf, preferSoundEndpoint: true);
  }

  /// 🔹 Upload EDF / âm thanh (multipart) — trả về Map kết quả JSON từ server
  ///
  /// - forceEdf: nếu true và filename thực trên device không có .edf, sẽ ép filename gửi lên server có đuôi .edf
  /// - preferSoundEndpoint: nếu true sẽ prefer '/upload_sound' endpoint for audio; otherwise '/upload' for EDF
  Future<Map<String, dynamic>> uploadFile(
    File file, {
    String? title,
    bool forceEdf = false,
    bool preferSoundEndpoint = false,
  }) async {
    final deviceId = await getDeviceId();

    final origBasename = p.basename(file.path);
    String basenameToSend = origBasename;
    final ext = p.extension(origBasename).toLowerCase();

    if (forceEdf && ext != '.edf') {
      basenameToSend = p.setExtension(origBasename, '.edf');
    }

    String endpoint;
    if ((ext == '.edf') || (forceEdf)) {
      endpoint = '/upload';
    } else {
      endpoint = preferSoundEndpoint ? '/upload_sound' : '/upload';
    }

    final uri = Uri.parse('$baseUrl$endpoint');
    final req = http.MultipartRequest('POST', uri);
    req.fields['device_id'] = deviceId;
    if (title != null) req.fields['title'] = title;

    final multipartFile = await http.MultipartFile.fromPath(
      'file',
      file.path,
      filename: basenameToSend,
    );
    req.files.add(multipartFile);

    final streamed = await req.send();
    final resp = await http.Response.fromStream(streamed);

    print('[uploadFile] POST $uri -> ${resp.statusCode}');
    print('[uploadFile] sent filename: $basenameToSend, body: ${resp.body}');

    if (resp.statusCode >= 400) {
      throw Exception('Upload failed: ${resp.statusCode} ${resp.body}');
    }

    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  /// 🔹 Gọi API /calculate (server phân tích EDF đã upload)
  Future<Map<String, dynamic>> calculate({
    required String filename,
    required String bedTime, // "HH:mm"
    String? wakeTime, // "HH:mm"
    String age = "30",
    String gender = "nam",
    String mode = "1",
  }) async {
    final uri = Uri.parse('$baseUrl/calculate');
    final body = jsonEncode({
      "filename": filename,
      "bed_time": bedTime,
      "wake_time": wakeTime ?? "06:30",
      "age": age,
      "gender": gender,
      "mode": mode,
    });

    final resp = await http.post(uri,
        headers: {'Content-Type': 'application/json'}, body: body);

    if (resp.statusCode != 200) {
      throw Exception('calculate failed: ${resp.statusCode} ${resp.body}');
    }

    return jsonDecode(resp.body) as Map<String, dynamic>;
  }
}
