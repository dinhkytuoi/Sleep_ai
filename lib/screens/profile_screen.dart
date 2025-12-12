// lib/screens/profile_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  // giữ nguyên các biến như yêu cầu
  final List<String> _genders = ['Male', 'Female'];
  String _selectedGender = 'Female';
  DateTime? _birthday;
  int _age = 18;
  final FlutterLocalNotificationsPlugin _localNotifPlugin =
      FlutterLocalNotificationsPlugin();

  // avatar managing
  String? _avatarPath;
  final ImagePicker _imagePicker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _initNotifications();
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final savedGender = prefs.getString('profile_gender');
    final savedBirthday = prefs.getString('profile_birthday');
    final savedAge = prefs.getInt('profile_age');
    final savedAvatar = prefs.getString('profile_avatar');
    if (!mounted) return;
    setState(() {
      if (savedGender != null) _selectedGender = savedGender;
      if (savedBirthday != null) {
        _birthday = DateTime.tryParse(savedBirthday);
      }
      if (savedAge != null) _age = savedAge;
      if (savedAvatar != null && savedAvatar.isNotEmpty)
        _avatarPath = savedAvatar;
    });
  }

  Future<void> _saveGender(String gender) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('profile_gender', gender);
  }

  Future<void> _saveBirthday(DateTime birth, int age) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('profile_birthday', birth.toIso8601String());
    await prefs.setInt('profile_age', age);
  }

  Future<void> _saveAvatar(String filePath) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('profile_avatar', filePath);
  }

  Future<void> _initNotifications() async {
    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const DarwinInitializationSettings iosSettings =
        DarwinInitializationSettings();
    final initSettings =
        InitializationSettings(android: androidSettings, iOS: iosSettings);
    await _localNotifPlugin.initialize(initSettings);
  }

  void _openGenderBottomSheet() {
    String tempSelected = _selectedGender;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.38,
          minChildSize: 0.2,
          maxChildSize: 0.6,
          builder: (context, scrollController) {
            return Container(
              decoration: const BoxDecoration(
                color: Color(0xFF111111),
                borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
              ),
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.white24,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Select Gender',
                    style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: Colors.white),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: ListView.separated(
                      controller: scrollController,
                      itemCount: _genders.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 6),
                      itemBuilder: (context, idx) {
                        final g = _genders[idx];
                        final isSelected = g == tempSelected;
                        return ListTile(
                          onTap: () {
                            setState(() {
                              tempSelected = g;
                            });
                          },
                          tileColor: isSelected
                              ? const Color(0xFF1E293B)
                              : Colors.transparent,
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12)),
                          title: Text(
                            g,
                            style: TextStyle(
                              fontSize: isSelected ? 18 : 16,
                              fontWeight: isSelected
                                  ? FontWeight.bold
                                  : FontWeight.w500,
                              color: isSelected ? Colors.white : Colors.white70,
                            ),
                          ),
                          trailing: isSelected
                              ? const Icon(Icons.check_circle,
                                  color: Color(0xFF3B82F6))
                              : null,
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          style: OutlinedButton.styleFrom(
                            backgroundColor: const Color(0xFF1B1B1B),
                            side: const BorderSide(color: Colors.white10),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                          ),
                          onPressed: () => Navigator.pop(ctx),
                          child: const Text('Cancel',
                              style: TextStyle(color: Colors.white70)),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF3B82F6),
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                          ),
                          onPressed: () async {
                            setState(() {
                              _selectedGender = tempSelected;
                            });
                            await _saveGender(_selectedGender);
                            if (!mounted) return;
                            final messenger = ScaffoldMessenger.of(context);
                            final body = 'Gender set to $_selectedGender';
                            messenger.showSnackBar(SnackBar(
                                content: Text(body),
                                duration: const Duration(seconds: 2)));
                            await _showSavedNotification(
                                'Profile updated', body);
                            if (!mounted) return;
                            Navigator.pop(ctx);
                          },
                          child: const Text('Save',
                              style: TextStyle(color: Colors.white)),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Future<void> _pickBirthday() async {
    final now = DateTime.now();
    final firstDate = DateTime(now.year - 100, now.month, now.day);
    final lastDate = now;
    final initialDate =
        _birthday ?? DateTime(now.year - 18, now.month, now.day);
    final picked = await showDatePicker(
      context: context,
      initialDate: initialDate,
      firstDate: firstDate,
      lastDate: lastDate,
      helpText: 'Select your birthday',
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.dark(
              primary: Color(0xFF3B82F6),
              onPrimary: Colors.white,
              surface: Color(0xFF222222),
              onSurface: Colors.white,
            ),
            dialogTheme: const DialogThemeData(
              backgroundColor: Color(0xFF111111),
            ),
          ),
          child: child ?? const SizedBox(),
        );
      },
    );
    if (!mounted) return;
    if (picked != null) {
      final computedAge = _computeAge(picked);
      setState(() {
        _birthday = picked;
        _age = computedAge;
      });
      await _saveBirthday(picked, computedAge);
      if (!mounted) return;
      final messenger = ScaffoldMessenger.of(context);
      final body = 'Birthday saved — age $_age';
      messenger.showSnackBar(
          SnackBar(content: Text(body), duration: const Duration(seconds: 2)));
      await _showSavedNotification('Profile updated', body);
    }
  }

  String get _birthdayText {
    if (_birthday == null) return '-';
    return DateFormat('yyyy-M-d').format(_birthday!);
  }

  Future<void> _showSavedNotification(String title, String body) async {
    const androidDetails = AndroidNotificationDetails(
      'profile_channel',
      'Profile updates',
      channelDescription: 'Notifications for profile changes',
      importance: Importance.defaultImportance,
      priority: Priority.defaultPriority,
    );
    const iosDetails = DarwinNotificationDetails();
    const platformDetails =
        NotificationDetails(android: androidDetails, iOS: iosDetails);
    await _localNotifPlugin.show(
      DateTime.now().millisecondsSinceEpoch.remainder(100000),
      title,
      body,
      platformDetails,
    );
  }

  int _computeAge(DateTime birthDate) {
    final now = DateTime.now();
    int age = now.year - birthDate.year;
    if (now.month < birthDate.month ||
        (now.month == birthDate.month && now.day < birthDate.day)) {
      age--;
    }
    return age;
  }

  void _openAvatarPicker() async {
    // chọn từ gallery, lưu vào thư mục app và lưu path vào prefs
    final ctx = context;
    showModalBottomSheet(
      context: ctx,
      backgroundColor: const Color(0xFF111111),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (sheetCtx) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.photo_library, color: Colors.white70),
                title: const Text('Choose from gallery',
                    style: TextStyle(color: Colors.white)),
                onTap: () async {
                  Navigator.pop(sheetCtx);
                  try {
                    final picked = await _imagePicker.pickImage(
                        source: ImageSource.gallery, imageQuality: 85);
                    if (picked == null) return;
                    final appDir = await getApplicationDocumentsDirectory();
                    final fileName = path.basename(picked.path);
                    final saved = await File(picked.path)
                        .copy('${appDir.path}/$fileName');
                    await _saveAvatar(saved.path);
                    if (!mounted) return;
                    setState(() => _avatarPath = saved.path);
                    ScaffoldMessenger.of(ctx).showSnackBar(
                        const SnackBar(content: Text('Avatar updated')));
                  } catch (e) {
                    if (!mounted) return;
                    ScaffoldMessenger.of(ctx).showSnackBar(
                        SnackBar(content: Text('Failed to pick image: $e')));
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.delete, color: Colors.white70),
                title: const Text('Remove avatar',
                    style: TextStyle(color: Colors.white)),
                onTap: () async {
                  Navigator.pop(sheetCtx);
                  if (_avatarPath != null) {
                    try {
                      final f = File(_avatarPath!);
                      if (await f.exists()) await f.delete();
                    } catch (_) {}
                    final prefs = await SharedPreferences.getInstance();
                    await prefs.remove('profile_avatar');
                    if (!mounted) return;
                    setState(() => _avatarPath = null);
                    ScaffoldMessenger.of(ctx).showSnackBar(
                        const SnackBar(content: Text('Avatar removed')));
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.close, color: Colors.white70),
                title: const Text('Cancel',
                    style: TextStyle(color: Colors.white70)),
                onTap: () => Navigator.pop(sheetCtx),
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    // Giao diện đẹp hơn: header với avatar, thông tin tóm tắt, dùng Card, ListTile
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/bg.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_ios_new,
                          color: Colors.white),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const SizedBox(width: 6),
                    const Text(
                      'My Profile',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const Spacer(),
                    // pencil icon removed per request
                  ],
                ),
                const SizedBox(height: 18),

                // Header Card
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.45),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.white10),
                  ),
                  child: Row(
                    children: [
                      // Avatar (clickable) — removed CircleAvatar, use clipped image or placeholder icon
                      InkWell(
                        onTap: _openAvatarPicker,
                        borderRadius: BorderRadius.circular(18),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(18),
                          child: Container(
                            width: 76,
                            height: 76,
                            decoration: BoxDecoration(
                              color: Colors.transparent,
                              borderRadius: BorderRadius.circular(18),
                              border: Border.all(color: Colors.white10),
                            ),
                            child: _avatarPath != null
                                ? Image.file(File(_avatarPath!),
                                    fit: BoxFit.cover, width: 76, height: 76)
                                : Container(
                                    decoration: const BoxDecoration(
                                      gradient: LinearGradient(
                                        begin: Alignment.topLeft,
                                        end: Alignment.bottomRight,
                                        colors: [
                                          Color(0xFF3B82F6),
                                          Color(0xFF60A5FA)
                                        ],
                                      ),
                                    ),
                                    child: const Center(
                                      child: Icon(Icons.person,
                                          size: 34, color: Colors.white),
                                    ),
                                  ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('Sleeper',
                                style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 18,
                                    fontWeight: FontWeight.w700)),
                            const SizedBox(height: 6),
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF1F2937),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Row(
                                    children: [
                                      const Icon(Icons.cake,
                                          size: 16, color: Colors.white70),
                                      const SizedBox(width: 6),
                                      Text(
                                          _birthday == null
                                              ? '-'
                                              : '$_age years',
                                          style: const TextStyle(
                                              color: Colors.white70,
                                              fontSize: 16)),
                                    ],
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF1F2937),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Row(
                                    children: [
                                      const Icon(Icons.male,
                                          size: 16, color: Colors.white70),
                                      const SizedBox(width: 6),
                                      Text(_selectedGender,
                                          style: const TextStyle(
                                              color: Colors.white70,
                                              fontSize: 16)),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 16),

                // Options Card
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.45),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.white10),
                  ),
                  child: Column(
                    children: [
                      // Gender row
                      ListTile(
                        onTap: _openGenderBottomSheet,
                        title: const Text('Gender',
                            style: TextStyle(
                                color: Colors.white70,
                                fontWeight: FontWeight.w600)),
                        trailing:
                            Row(mainAxisSize: MainAxisSize.min, children: [
                          Text(_selectedGender,
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w600,
                                  fontSize: 16)),
                          const SizedBox(width: 8),
                          const Icon(Icons.chevron_right,
                              color: Colors.white70),
                        ]),
                      ),
                      Divider(color: Colors.white12, height: 1),
                      // Age row
                      ListTile(
                        onTap: _pickBirthday,
                        title: const Text('Age',
                            style: TextStyle(
                                color: Colors.white70,
                                fontWeight: FontWeight.w600)),
                        trailing:
                            Row(mainAxisSize: MainAxisSize.min, children: [
                          Text(_birthday == null ? '-' : '$_age',
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w600,
                                  fontSize: 16)),
                          const SizedBox(width: 8),
                          const Icon(Icons.chevron_right,
                              color: Colors.white70),
                        ]),
                      ),
                    ],
                  ),
                ),

                // Birthday text
                if (_birthday != null) ...[
                  const SizedBox(height: 12),
                  Text('Birthday: $_birthdayText',
                      style: const TextStyle(color: Colors.white54)),
                ],

                const Spacer(),

                // Action buttons area
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () {
                          // reset demo (you can keep as placeholder)
                        },
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: Colors.white10),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12)),
                        ),
                        child: const Text('Reset',
                            style: TextStyle(color: Colors.white70)),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton(
                        // disabled because save not used yet
                        onPressed: null,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF3B82F6),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12)),
                        ),
                        child: const Text('Save profile',
                            style: TextStyle(color: Colors.white)),
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
