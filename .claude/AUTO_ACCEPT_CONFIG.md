# Claude Code Auto-Accept Configuration

## Overview
This configuration enables automatic acceptance of file edits and other operations without user prompts.

## Configuration Location
- **File**: `.claude/settings.local.json`
- **Type**: Local settings (not committed to version control)

## Auto-Accepted Operations

The following tools are configured to run without user confirmation:

### Build & Test Operations
- `Bash(cargo test:*)` - All cargo test commands
- `Bash(cargo build:*)` - All cargo build commands

### File System Operations
- `Bash(ls:*)` - List directory contents
- `Bash(du:*)` - Check disk usage
- `Bash(awk:*)` - AWK text processing
- `Bash(wc:*)` - Word/line counting

### File Editing Operations
- `Edit` - Modify existing files
- `Write` - Create/overwrite files
- `NotebookEdit` - Edit Jupyter notebooks
- `TodoWrite` - Manage todo lists

## How It Works

Claude Code checks the `permissions` section in settings:
- **allow**: Operations that execute automatically without prompts
- **deny**: Operations that are blocked
- **ask**: Operations that require user confirmation (default for unlisted)

## Modifying Configuration

To add more auto-accepted operations:
1. Edit `.claude/settings.local.json`
2. Add tool names to the `"allow"` array
3. Use patterns like `"Bash(command:*)"` for specific bash commands
4. Save the file - changes take effect immediately

## Safety Considerations

While auto-accept improves workflow speed, consider:
- Review changes with version control before committing
- Use `deny` list for sensitive operations
- Keep backups of important files
- Monitor Claude's actions in the terminal

## Reverting Changes

To disable auto-accept for any operation:
1. Remove the tool name from the `"allow"` array
2. Optionally add to `"ask"` array for explicit prompting
3. Or add to `"deny"` array to block completely

## Example: Disable Auto-Edit
```json
{
  "permissions": {
    "allow": [
      // Remove "Edit" from here
    ],
    "ask": [
      "Edit"  // Add here to require confirmation
    ]
  }
}
```