import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(const HallucinationLabelerApp());
}

class HallucinationLabelerApp extends StatelessWidget {
  const HallucinationLabelerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hallucination Labeler',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1F6FEB),
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F6FB),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: BorderSide(color: Colors.grey.shade200),
          ),
        ),
      ),
      home: const LabelerScreen(),
    );
  }
}

class LabelerScreen extends StatefulWidget {
  const LabelerScreen({super.key});

  @override
  State<LabelerScreen> createState() => _LabelerScreenState();
}

class _LabelerScreenState extends State<LabelerScreen> {
  // API Configuration
  static const String apiBase = 'http://localhost:3000';

  // Hallucination types
  static const List<String> hallucinationTypes = [
    'Factual Inconsistency',
    'Source Mismatch',
    'Fabricated Information',
    'Context Misinterpretation',
    'Temporal Inconsistency',
    'Logical Contradiction',
    'Overgeneralization',
    'Unsupported Claim',
  ];

  // State
  List<Map<String, dynamic>> records = [];
  List<int> filteredIndices = [];
  int currentIndex = 0;
  String? selectedFilename;
  List<String> availableFiles = [];
  bool isServerConnected = false;
  bool isSaving = false;
  bool isDirty = false;
  String filterMode = 'all';
  String searchQuery = '';
  Timer? saveTimer;
  String? lastError;

  // Controllers
  final TextEditingController searchController = TextEditingController();
  final TextEditingController reasonController = TextEditingController();
  final FocusNode mainFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _checkServerAndLoadFiles();
    _loadLastFile();
  }

  @override
  void dispose() {
    searchController.dispose();
    reasonController.dispose();
    mainFocusNode.dispose();
    saveTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadLastFile() async {
    final prefs = await SharedPreferences.getInstance();
    final lastFile = prefs.getString('last_file');
    if (lastFile != null && lastFile.isNotEmpty) {
      setState(() {
        selectedFilename = lastFile;
      });
      await _loadFile(lastFile);
    }
  }

  Future<void> _saveLastFile(String filename) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('last_file', filename);
  }

  Future<void> _checkServerAndLoadFiles() async {
    try {
      final response = await http
          .get(Uri.parse('$apiBase/api/health'))
          .timeout(const Duration(seconds: 3));

      setState(() {
        isServerConnected = response.statusCode == 200;
      });

      if (isServerConnected) {
        await _loadFileList();
      }
    } catch (e) {
      setState(() {
        isServerConnected = false;
        lastError = 'Server offline';
      });
    }
  }

  Future<void> _loadFileList() async {
    try {
      final response = await http.get(Uri.parse('$apiBase/api/files'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final files = List<String>.from(data['files']).toSet().toList()..sort();
        setState(() {
          availableFiles = files;
          if (selectedFilename != null &&
              !availableFiles.contains(selectedFilename)) {
            selectedFilename = null;
          }
        });
      }
    } catch (e) {
      _showMessage('Failed to load file list: $e');
    }
  }

  Future<void> _loadFile(String filename) async {
    try {
      final response = await http.get(
        Uri.parse('$apiBase/api/load/${Uri.encodeComponent(filename)}'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          records = List<Map<String, dynamic>>.from(
            (data['records'] as List).map((r) => Map<String, dynamic>.from(r)),
          );
          selectedFilename = filename;
          currentIndex = 0;
          isDirty = false;
          lastError = null;
        });
        _applyFilter();
        await _saveLastFile(filename);
        _showMessage('Loaded ${records.length} records');
      }
    } catch (e) {
      _showMessage('Failed to load file: $e');
    }
  }

  Future<void> _saveToServer() async {
    if (selectedFilename == null || !isServerConnected || isSaving) {
      return;
    }

    setState(() {
      isSaving = true;
    });

    try {
      final response = await http.post(
        Uri.parse(
          '$apiBase/api/save/${Uri.encodeComponent(selectedFilename!)}',
        ),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'records': records}),
      );

      if (response.statusCode == 200) {
        setState(() {
          isDirty = false;
          lastError = null;
        });
        _showMessage('Autosaved');
      } else {
        throw Exception('Save failed: ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        lastError = 'Save failed: $e';
      });
    } finally {
      setState(() {
        isSaving = false;
      });
    }
  }

  void _scheduleAutoSave() {
    saveTimer?.cancel();
    saveTimer = Timer(const Duration(milliseconds: 450), () {
      _saveToServer();
    });
  }

  void _markDirty() {
    setState(() {
      isDirty = true;
      lastError = null;
    });
    _scheduleAutoSave();
  }

  void _applyFilter() {
    filteredIndices.clear();

    for (int i = 0; i < records.length; i++) {
      bool matches = true;

      // Apply filter mode
      if (filterMode == 'hall') {
        matches = records[i]['human_label'] == 1;
      } else if (filterMode == 'ok') {
        matches = records[i]['human_label'] == 0;
      } else if (filterMode == 'unlabeled') {
        matches =
            records[i]['human_label'] != 0 && records[i]['human_label'] != 1;
      }

      // Apply search
      if (matches && searchQuery.isNotEmpty) {
        matches = json
            .encode(records[i])
            .toLowerCase()
            .contains(searchQuery.toLowerCase());
      }

      if (matches) {
        filteredIndices.add(i);
      }
    }

    if (currentIndex >= filteredIndices.length) {
      currentIndex = filteredIndices.isEmpty ? 0 : filteredIndices.length - 1;
    }

    setState(() {});
  }

  Map<String, dynamic>? get currentRecord {
    if (filteredIndices.isEmpty || currentIndex >= filteredIndices.length) {
      return null;
    }
    return records[filteredIndices[currentIndex]];
  }

  void _setLabel(int value) {
    final record = currentRecord;
    if (record != null) {
      record['human_label'] = value;
      _markDirty();
      _applyFilter();
    }
  }

  void _removeLabel() {
    final record = currentRecord;
    if (record != null) {
      record.remove('human_label');
      record.remove('hallucination_type');
      record.remove('reason');
      _markDirty();
      _applyFilter();
    }
  }

  void _navigate(int delta) {
    if (filteredIndices.isNotEmpty) {
      setState(() {
        currentIndex = (currentIndex + delta).clamp(
          0,
          filteredIndices.length - 1,
        );
      });
    }
  }

  void _showMessage(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          duration: const Duration(seconds: 2),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  int get labeledCount {
    return records
        .where((r) => r['human_label'] == 0 || r['human_label'] == 1)
        .length;
  }

  @override
  Widget build(BuildContext context) {
    return KeyboardListener(
      focusNode: mainFocusNode,
      autofocus: true,
      onKeyEvent: (event) {
        if (event is KeyDownEvent) {
          if (event.logicalKey == LogicalKeyboardKey.keyH) {
            _setLabel(1);
          } else if (event.logicalKey == LogicalKeyboardKey.keyN) {
            _setLabel(0);
          } else if (event.logicalKey == LogicalKeyboardKey.keyU) {
            _removeLabel();
          } else if (event.logicalKey == LogicalKeyboardKey.arrowLeft) {
            _navigate(-1);
          } else if (event.logicalKey == LogicalKeyboardKey.arrowRight) {
            _navigate(1);
          }
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Hallucination Labeler'),
          backgroundColor: Colors.white,
          foregroundColor: Colors.black87,
          elevation: 0,
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(1),
            child: Container(color: Colors.grey.shade200, height: 1),
          ),
          actions: [
            // Server status
            Container(
              margin: const EdgeInsets.only(right: 16),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: isServerConnected
                    ? const Color(0xFFDFF6E8)
                    : const Color(0xFFFDE7E5),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: isServerConnected
                          ? const Color(0xFF157347)
                          : const Color(0xFFB42318),
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    isServerConnected ? 'Online' : 'Offline',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: isServerConnected
                          ? const Color(0xFF157347)
                          : const Color(0xFFB42318),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        body: Column(
          children: [
            // Toolbar
            Container(
              padding: const EdgeInsets.all(16),
              color: Colors.white,
              child: Column(
                children: [
                  Row(
                    children: [
                      // File selector
                      Expanded(
                        child: DropdownButtonFormField<String>(
                          value: availableFiles.contains(selectedFilename)
                              ? selectedFilename
                              : null,
                          decoration: const InputDecoration(
                            labelText: 'Select File',
                            border: OutlineInputBorder(),
                            contentPadding: EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 8,
                            ),
                          ),
                          items: availableFiles.map((file) {
                            return DropdownMenuItem(
                              value: file,
                              child: Text(
                                file,
                                overflow: TextOverflow.ellipsis,
                              ),
                            );
                          }).toList(),
                          onChanged: (value) {
                            if (value != null) {
                              _loadFile(value);
                            }
                          },
                        ),
                      ),
                      const SizedBox(width: 12),
                      // Refresh button
                      IconButton(
                        onPressed: _checkServerAndLoadFiles,
                        icon: const Icon(Icons.refresh),
                        tooltip: 'Refresh file list',
                      ),
                      // Save button
                      IconButton(
                        onPressed: _saveToServer,
                        icon: const Icon(Icons.save),
                        tooltip: 'Save now',
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  // Status bar
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          _buildStatusText(),
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey.shade700,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  // Progress bar
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value:
                          records.isEmpty ? 0 : labeledCount / records.length,
                      minHeight: 6,
                      backgroundColor: Colors.grey.shade200,
                      valueColor: const AlwaysStoppedAnimation<Color>(
                        Color(0xFF1F6FEB),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Main content
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Sidebar
                  SizedBox(
                    width: 320,
                    child: Column(
                      children: [
                        // Filters and search
                        Container(
                          padding: const EdgeInsets.all(12),
                          color: Colors.white,
                          child: Column(
                            children: [
                              TextField(
                                controller: searchController,
                                decoration: const InputDecoration(
                                  hintText: 'Search...',
                                  prefixIcon: Icon(Icons.search),
                                  border: OutlineInputBorder(),
                                  contentPadding: EdgeInsets.symmetric(
                                    horizontal: 12,
                                    vertical: 8,
                                  ),
                                ),
                                onChanged: (value) {
                                  setState(() {
                                    searchQuery = value;
                                  });
                                  _applyFilter();
                                },
                              ),
                              const SizedBox(height: 8),
                              Wrap(
                                spacing: 8,
                                children: [
                                  FilterChip(
                                    label: const Text('All'),
                                    selected: filterMode == 'all',
                                    onSelected: (selected) {
                                      setState(() {
                                        filterMode = 'all';
                                      });
                                      _applyFilter();
                                    },
                                  ),
                                  FilterChip(
                                    label: const Text('Hallucinated'),
                                    selected: filterMode == 'hall',
                                    onSelected: (selected) {
                                      setState(() {
                                        filterMode = 'hall';
                                      });
                                      _applyFilter();
                                    },
                                  ),
                                  FilterChip(
                                    label: const Text('Correct'),
                                    selected: filterMode == 'ok',
                                    onSelected: (selected) {
                                      setState(() {
                                        filterMode = 'ok';
                                      });
                                      _applyFilter();
                                    },
                                  ),
                                  FilterChip(
                                    label: const Text('Unlabeled'),
                                    selected: filterMode == 'unlabeled',
                                    onSelected: (selected) {
                                      setState(() {
                                        filterMode = 'unlabeled';
                                      });
                                      _applyFilter();
                                    },
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),

                        // Record list
                        Expanded(
                          child: ListView.builder(
                            itemCount: filteredIndices.length,
                            itemBuilder: (context, index) {
                              final recordIndex = filteredIndices[index];
                              final record = records[recordIndex];
                              final isActive = index == currentIndex;

                              return InkWell(
                                onTap: () {
                                  setState(() {
                                    currentIndex = index;
                                  });
                                },
                                child: Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    color: isActive
                                        ? const Color(0xFFEFF6FF)
                                        : Colors.white,
                                    border: Border(
                                      bottom: BorderSide(
                                        color: Colors.grey.shade200,
                                      ),
                                      left: BorderSide(
                                        color: isActive
                                            ? const Color(0xFF1F6FEB)
                                            : Colors.transparent,
                                        width: 3,
                                      ),
                                    ),
                                  ),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        _truncate(
                                          record['question'] ??
                                              record['id'] ??
                                              'Untitled',
                                          90,
                                        ),
                                        style: TextStyle(
                                          fontSize: 13,
                                          fontWeight: isActive
                                              ? FontWeight.w600
                                              : FontWeight.normal,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      _buildStatusChip(record),
                                    ],
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Divider
                  Container(width: 1, color: Colors.grey.shade200),

                  // Detail view
                  Expanded(child: _buildDetailView()),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _buildStatusText() {
    final parts = <String>[];

    if (selectedFilename != null) {
      parts.add(selectedFilename!);
    } else {
      parts.add('No file loaded');
    }

    if (records.isNotEmpty) {
      parts.add('${records.length} records');
      parts.add('$labeledCount labeled');
    }

    if (isServerConnected && selectedFilename != null) {
      parts.add('autosave enabled');
    }

    if (isSaving) {
      parts.add('saving...');
    } else if (isDirty) {
      parts.add('unsaved changes');
    } else if (records.isNotEmpty) {
      parts.add('saved');
    }

    if (lastError != null) {
      parts.add(lastError!);
    }

    return parts.join(' | ');
  }

  Widget _buildStatusChip(Map<String, dynamic> record) {
    String text;
    Color bgColor;
    Color textColor;

    if (record['human_label'] == 1) {
      text = 'Hallucinated';
      bgColor = const Color(0xFFFDE7E5);
      textColor = const Color(0xFFB42318);
    } else if (record['human_label'] == 0) {
      text = 'Correct';
      bgColor = const Color(0xFFDFF6E8);
      textColor = const Color(0xFF157347);
    } else {
      text = 'Unlabeled';
      bgColor = Colors.grey.shade100;
      textColor = Colors.grey.shade700;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: textColor,
        ),
      ),
    );
  }

  Widget _buildDetailView() {
    final record = currentRecord;

    if (record == null) {
      return Center(
        child: Card(
          margin: const EdgeInsets.all(24),
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.description_outlined,
                  size: 64,
                  color: Colors.grey.shade400,
                ),
                const SizedBox(height: 16),
                Text(
                  records.isEmpty
                      ? 'Select a file to start labeling'
                      : 'No records match the filter',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 24),
                const Text('Keyboard Shortcuts:'),
                const SizedBox(height: 8),
                _buildShortcut('H', 'Mark hallucinated'),
                _buildShortcut('N', 'Mark correct'),
                _buildShortcut('U', 'Remove label'),
                _buildShortcut('← / →', 'Navigate records'),
              ],
            ),
          ),
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Label badge
          _buildLabelBadge(record),

          const SizedBox(height: 16),

          // Action buttons
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ElevatedButton.icon(
                onPressed: () => _setLabel(1),
                icon: const Icon(Icons.error_outline),
                label: const Text('Hallucinated (H)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFB42318),
                  foregroundColor: Colors.white,
                ),
              ),
              ElevatedButton.icon(
                onPressed: () => _setLabel(0),
                icon: const Icon(Icons.check_circle_outline),
                label: const Text('Correct (N)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF157347),
                  foregroundColor: Colors.white,
                ),
              ),
              OutlinedButton.icon(
                onPressed: _removeLabel,
                icon: const Icon(Icons.remove_circle_outline),
                label: const Text('Remove Label (U)'),
              ),
              OutlinedButton.icon(
                onPressed: () {
                  Clipboard.setData(
                    ClipboardData(
                      text: const JsonEncoder.withIndent('  ').convert(record),
                    ),
                  );
                  _showMessage('Copied to clipboard');
                },
                icon: const Icon(Icons.copy),
                label: const Text('Copy JSON'),
              ),
            ],
          ),

          const SizedBox(height: 16),

          // Hallucination type dropdown
          if (record['human_label'] == 1) ...[
            DropdownButtonFormField<String>(
              value: (record['hallucination_type'] is String &&
                      hallucinationTypes.contains(record['hallucination_type']))
                  ? record['hallucination_type'] as String
                  : null,
              decoration: const InputDecoration(
                labelText: 'Hallucination Type',
                border: OutlineInputBorder(),
              ),
              items: [
                const DropdownMenuItem(
                  value: null,
                  child: Text('Select type...'),
                ),
                ...hallucinationTypes.map((type) {
                  return DropdownMenuItem(value: type, child: Text(type));
                }),
              ],
              onChanged: (value) {
                record['hallucination_type'] = value;
                _markDirty();
              },
            ),
            const SizedBox(height: 12),
            TextField(
              controller: reasonController..text = record['reason'] ?? '',
              decoration: const InputDecoration(
                labelText: 'Reason (optional)',
                border: OutlineInputBorder(),
                hintText: 'Explain why this is hallucinated...',
              ),
              maxLines: 3,
              onChanged: (value) {
                if (value.trim().isEmpty) {
                  record.remove('reason');
                } else {
                  record['reason'] = value;
                }
                _markDirty();
              },
            ),
            const SizedBox(height: 16),
          ],

          // Navigation
          Row(
            children: [
              IconButton(
                onPressed: () {
                  setState(() {
                    currentIndex = 0;
                  });
                },
                icon: const Icon(Icons.first_page),
                tooltip: 'First',
              ),
              IconButton(
                onPressed: () => _navigate(-1),
                icon: const Icon(Icons.chevron_left),
                tooltip: 'Previous',
              ),
              Text(
                '${currentIndex + 1} / ${filteredIndices.length}',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
              IconButton(
                onPressed: () => _navigate(1),
                icon: const Icon(Icons.chevron_right),
                tooltip: 'Next',
              ),
              IconButton(
                onPressed: () {
                  setState(() {
                    currentIndex = filteredIndices.length - 1;
                  });
                },
                icon: const Icon(Icons.last_page),
                tooltip: 'Last',
              ),
            ],
          ),

          const SizedBox(height: 24),

          // Record fields
          ...record.entries
              .where(
            (e) => ![
              'human_label',
              'hallucination_type',
              'reason',
            ].contains(e.key),
          )
              .map((entry) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        entry.key,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFF1F6FEB),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _formatValue(entry.value),
                        style: const TextStyle(fontSize: 14),
                      ),
                    ],
                  ),
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildLabelBadge(Map<String, dynamic> record) {
    String text;
    Color bgColor;
    Color textColor;

    if (record['human_label'] == 1) {
      text = 'Hallucinated';
      bgColor = const Color(0xFFFDE7E5);
      textColor = const Color(0xFFB42318);
    } else if (record['human_label'] == 0) {
      text = 'Correct';
      bgColor = const Color(0xFFDFF6E8);
      textColor = const Color(0xFF157347);
    } else {
      text = 'Unlabeled';
      bgColor = Colors.grey.shade100;
      textColor = Colors.grey.shade700;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: textColor,
        ),
      ),
    );
  }

  Widget _buildShortcut(String key, String description) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.grey.shade200,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              key,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
            ),
          ),
          const SizedBox(width: 8),
          Text(description, style: const TextStyle(fontSize: 12)),
        ],
      ),
    );
  }

  String _truncate(String text, int maxLength) {
    if (text.length <= maxLength) return text;
    return '${text.substring(0, maxLength)}...';
  }

  String _formatValue(dynamic value) {
    if (value is Map || value is List) {
      return const JsonEncoder.withIndent('  ').convert(value);
    }
    return value.toString();
  }
}
