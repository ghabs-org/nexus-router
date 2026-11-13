#!/usr/bin/env node
// Keep the router in Python; npm distributes a small, argument-safe bootstrapper.
import { spawnSync } from 'node:child_process';
import { mkdirSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const version = '0.2.0';
const args = process.argv.slice(2);
const executable = process.env.NEXUS_GATEWAY_EXECUTABLE;
function run(command, parameters) {
  const result = spawnSync(command, parameters, { stdio: 'inherit', shell: false });
  if (result.error) {
    console.error(`Cannot launch ${command}: ${result.error.message}`);
    process.exit(1);
  }
  return result.status ?? 1;
}
if (executable) process.exit(run(executable, args));
const root = process.env.NEXUS_GATEWAY_RUNTIME_DIR ?? join(homedir(), '.local', 'share', 'nexus-router-gateway', version);
const python = process.env.NEXUS_GATEWAY_PYTHON ?? 'python3';
const windows = process.platform === 'win32';
const pythonInVenv = join(root, windows ? 'Scripts' : 'bin', windows ? 'python.exe' : 'python');
const entry = join(root, windows ? 'Scripts' : 'bin', windows ? 'nexus-router-gateway.exe' : 'nexus-router-gateway');
if (!existsSync(entry)) {
  const check = spawnSync(python, ['-c', 'import sys; raise SystemExit(sys.version_info < (3, 11))'], { shell: false });
  if (check.error || check.status !== 0) {
    console.error('Python 3.11+ is required. Set NEXUS_GATEWAY_PYTHON to its executable.');
    process.exit(1);
  }
  mkdirSync(root, { recursive: true, mode: 0o700 });
  if (!existsSync(pythonInVenv) && run(python, ['-m', 'venv', root]) !== 0) process.exit(1);
  const source = process.env.NEXUS_GATEWAY_WHEEL ?? `nexus-router-gateway==${version}`;
  if (run(pythonInVenv, ['-m', 'pip', 'install', source]) !== 0) process.exit(1);
}
process.exit(run(entry, args));
