// lib/screens/sound_detail_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import '../services/sound_manager.dart';

class SoundDetailScreen extends StatefulWidget {
  final String title;
  final String imageAsset;
  final String audioAsset; // e.g. 'assets/audio/drizzling.mp3'

  const SoundDetailScreen({
    super.key,
    required this.title,
    required this.imageAsset,
    required this.audioAsset,
  });

  @override
  State<SoundDetailScreen> createState() => _SoundDetailScreenState();
}

class _SoundDetailScreenState extends State<SoundDetailScreen> {
  final AudioPlayer _player = AudioPlayer();
  bool _isPlaying = false;
  Duration _duration = Duration.zero;
  Duration _position = Duration.zero;
  bool _isDownloading = false;
  String? _localPath;

  @override
  void initState() {
    super.initState();
    _initLocal();
    _setupPlayer();
  }

  Future<void> _initLocal() async {
    final path = await SoundManager.getLocalPath(widget.audioAsset);
    if (mounted) setState(() => _localPath = path);
  }

  void _setupPlayer() {
    _player.onPlayerComplete.listen((_) {
      setState(() {
        _isPlaying = false;
        _position = _duration;
      });
    });
    _player.onDurationChanged.listen((d) {
      setState(() => _duration = d);
    });
    _player.onPositionChanged.listen((p) {
      setState(() => _position = p);
    });
  }

  Future<void> _play() async {
    if (_localPath != null) {
      await _player.play(DeviceFileSource(_localPath!));
    } else {
      // play from asset: AssetSource expects path relative to pubspec assets
      // For an asset path like 'assets/audio/drizzling.mp3', pass 'audio/drizzling.mp3'
      final rel = widget.audioAsset.replaceFirst('assets/', '');
      await _player.play(AssetSource(rel));
    }
    setState(() => _isPlaying = true);
  }

  Future<void> _pause() async {
    await _player.pause();
    setState(() => _isPlaying = false);
  }

  Future<void> _seekTo(Duration pos) async {
    await _player.seek(pos);
  }

  String _fmt(Duration d) {
    final mm = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final ss = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$mm:$ss';
  }

  Future<void> _downloadToAppFolder() async {
    setState(() => _isDownloading = true);
    try {
      final fileName = widget.audioAsset.split('/').last;
      final bytes =
          (await rootBundle.load(widget.audioAsset)).buffer.asUint8List();

      final appDoc = await getApplicationDocumentsDirectory();
      final soundsDir = Directory('${appDoc.path}/sounds');
      if (!await soundsDir.exists()) await soundsDir.create(recursive: true);

      final dest = File('${soundsDir.path}/$fileName');
      await dest.writeAsBytes(bytes);

      await SoundManager.addDownloaded(widget.audioAsset, dest.path);
      if (mounted) {
        setState(() {
          _localPath = dest.path;
          _isDownloading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to app folder: ${dest.path}')),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isDownloading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Download failed: $e')),
        );
      }
    }
  }

  Future<void> _deleteLocal() async {
    if (_localPath == null) return;
    try {
      final f = File(_localPath!);
      if (await f.exists()) await f.delete();
      await SoundManager.removeDownloaded(widget.audioAsset);
      if (mounted) {
        setState(() => _localPath = null);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Removed from local storage')),
        );
      }
    } catch (e) {
      if (mounted)
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Delete failed: $e')));
    }
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final pos = _position.inSeconds > 0 ? _fmt(_position) : '00:00';
    final tot = _duration.inSeconds > 0 ? _fmt(_duration) : '--:--';

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close_rounded),
          onPressed: () {
            _player.stop();
            Navigator.pop(context);
          },
        ),
      ),
      extendBodyBehindAppBar: true,
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(widget.imageAsset,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(color: Colors.black12)),
          ),
          Positioned.fill(
              child: Container(color: Colors.black.withOpacity(0.45))),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18.0),
              child: Column(
                children: [
                  const SizedBox(height: 80),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Image.asset(widget.imageAsset,
                        width: double.infinity, height: 220, fit: BoxFit.cover),
                  ),
                  const SizedBox(height: 16),
                  Text(widget.title,
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold)),
                  const SizedBox(height: 12),
                  Slider(
                    min: 0,
                    max: (_duration.inMilliseconds > 0)
                        ? _duration.inMilliseconds.toDouble()
                        : 1.0,
                    value: (_position.inMilliseconds > 0)
                        ? _position.inMilliseconds.toDouble().clamp(
                            0,
                            (_duration.inMilliseconds > 0)
                                ? _duration.inMilliseconds.toDouble()
                                : 1.0)
                        : 0.0,
                    onChanged: (v) {
                      final newPos = Duration(milliseconds: v.toInt());
                      _seekTo(newPos);
                    },
                  ),
                  Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(pos,
                            style: const TextStyle(color: Colors.white70)),
                        Text(tot,
                            style: const TextStyle(color: Colors.white70)),
                      ]),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      IconButton(
                        iconSize: 52,
                        color: Colors.white,
                        onPressed: () async {
                          if (_isPlaying) {
                            await _pause();
                          } else {
                            await _play();
                          }
                        },
                        icon: Icon(_isPlaying
                            ? Icons.pause_circle_filled
                            : Icons.play_circle_filled),
                      ),
                      const SizedBox(width: 20),
                      _localPath == null
                          ? ElevatedButton.icon(
                              onPressed:
                                  _isDownloading ? null : _downloadToAppFolder,
                              icon: _isDownloading
                                  ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(
                                          strokeWidth: 2, color: Colors.white))
                                  : const Icon(Icons.download_rounded),
                              label: const Text('Download'),
                            )
                          : ElevatedButton.icon(
                              onPressed: _deleteLocal,
                              icon: const Icon(Icons.delete_outline),
                              label: const Text('Remove'),
                              style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.redAccent),
                            ),
                    ],
                  ),
                  const Spacer(),
                ],
              ),
            ),
          )
        ],
      ),
    );
  }
}
