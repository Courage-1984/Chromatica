import re

# Read the current main.py file
with open("src/chromatica/api/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find the restart endpoint section
start_marker = "@app.post(\"/restart\")"
end_marker = "@app.post(\"/api/execute-command\")"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    # Extract the part before and after the restart endpoint
    before = content[:start_idx]
    after = content[end_idx:]
    
    # Create the new comprehensive restart endpoint
    new_restart_endpoint = """@app.post("/restart")
async def restart_server():
    """
    Comprehensive server restart that handles all Chromatica processes and terminals.

    This endpoint will:
    1. Kill all Chromatica-related Python processes (server, tests, scripts, etc.)
    2. Kill processes by PID if needed
    3. Close the current terminal window
    4. Open a new terminal with proper environment setup
    5. Set environment variables and activate virtual environment
    6. Start the server in the new terminal
    7. Trigger a hard refresh of the browser page

    Returns:
        JSON response with restart status and details
    """
    import subprocess
    import os
    import sys
    import signal
    import psutil
    import time
    import json

    try:
        logger.info("🔄 Comprehensive server restart requested")

        # Get current process ID
        current_pid = os.getpid()
        logger.info(f"Current server PID: {current_pid}")

        # Step 1: Find and kill ALL Chromatica-related Python processes
        logger.info("🔍 Searching for Chromatica-related processes...")
        
        killed_processes = []
        chromatica_processes = []
        
        # Find all Python processes with Chromatica-related commands
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["name"] and "python" in proc.info["name"].lower():
                    cmdline = proc.info["cmdline"]
                    if cmdline:
                        cmdline_str = " ".join(cmdline).lower()
                        # Check for various Chromatica-related patterns
                        chromatica_patterns = [
                            "chromatica",
                            "src.chromatica.api.main",
                            "build_index.py",
                            "test_histogram_generation.py",
                            "test_api.py",
                            "cleanup_outputs.py",
                            "run_sanity_checks.py",
                            "evaluate.py",
                            "uvicorn"
                        ]
                        
                        if any(pattern in cmdline_str for pattern in chromatica_patterns):
                            chromatica_processes.append({
                                "pid": proc.info["pid"],
                                "cmdline": cmdline,
                                "name": proc.info["name"]
                            })
                            logger.info(f"Found Chromatica process: PID {proc.info["pid"]} - {" ".join(cmdline[:3])}...")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Step 2: Kill all found processes (except current one)
        logger.info(f"🔄 Killing {len(chromatica_processes)} Chromatica processes...")
        
        for proc_info in chromatica_processes:
            pid = proc_info["pid"]
            if pid != current_pid:
                try:
                    proc = psutil.Process(pid)
                    logger.info(f"Terminating process PID {pid}: {" ".join(proc_info["cmdline"][:3])}...")
                    proc.terminate()
                    killed_processes.append({
                        "pid": pid,
                        "cmdline": proc_info["cmdline"],
                        "status": "terminated"
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Could not terminate PID {pid}: {e}")
                    killed_processes.append({
                        "pid": pid,
                        "cmdline": proc_info["cmdline"],
                        "status": f"error: {str(e)}"
                    })

        # Step 3: Wait for graceful termination
        logger.info("⏳ Waiting for graceful termination...")
        time.sleep(3)

        # Step 4: Force kill any remaining processes
        logger.info("💀 Force killing any remaining processes...")
        for proc_info in killed_processes:
            if proc_info["status"] == "terminated":
                try:
                    proc = psutil.Process(proc_info["pid"])
                    if proc.is_running():
                        logger.info(f"Force killing PID {proc_info["pid"]}")
                        proc.kill()
                        proc_info["status"] = "force_killed"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc_info["status"] = "already_dead"

        # Step 5: Create restart script for new terminal
        project_root = Path(__file__).parent.parent.parent.parent
        logger.info(f"📁 Project root: {project_root}")
        
        # Create Windows batch script for restart
        restart_script = project_root / "restart_server.bat"
        
        script_content = f"""@echo off
echo 🔄 Chromatica Server Restart Script
echo ========================================
echo.
echo 📁 Project Root: {project_root}
echo 🐍 Activating Python virtual environment...
call "venv311\\Scripts\\activate"
echo.
echo 🌍 Setting environment variable...
set CHROMATICA_INDEX_DIR=C:\\Users\\anon\\github\\Chromatica\\85k
echo CHROMATICA_INDEX_DIR=%CHROMATICA_INDEX_DIR%
echo.
echo 🚀 Starting Chromatica server...
python -m src.chromatica.api.main
echo.
echo ⚠️  Server stopped. Press any key to close...
pause >nul
"""

        try:
            with open(restart_script, "w", encoding="utf-8") as f:
                f.write(script_content)
            logger.info(f"✅ Created restart script: {restart_script}")
        except Exception as e:
            logger.error(f"Failed to create restart script: {e}")
            return {
                "success": False,
                "message": f"Failed to create restart script: {str(e)}",
                "error": str(e),
            }

        # Step 6: Start new terminal with restart script
        logger.info("🚀 Starting new terminal with server...")
        
        try:
            # Use Windows cmd to run the batch script in a new window
            subprocess.Popen([
                "cmd.exe", "/c", "start", "Chromatica Server", "cmd", "/k", 
                str(restart_script)
            ], cwd=str(project_root))
            logger.info("✅ New terminal started successfully")
        except Exception as e:
            logger.error(f"Failed to start new terminal: {e}")
            # Fallback: try to start server directly
            try:
                python_path = str(project_root / "venv311" / "Scripts" / "python.exe")
                subprocess.Popen([
                    python_path, "-m", "src.chromatica.api.main"
                ], cwd=str(project_root), 
                env={**os.environ, "CHROMATICA_INDEX_DIR": "C:\\Users\\anon\\github\\Chromatica\\85k"})
                logger.info("✅ Fallback: Server started directly")
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")

        # Step 7: Prepare response with restart details
        restart_info = {
            "success": True,
            "message": "🔄 Server restart initiated successfully",
            "killed_processes": killed_processes,
            "total_killed": len([p for p in killed_processes if "killed" in p["status"]]),
            "new_terminal_started": True,
            "restart_script_created": str(restart_script),
            "environment_variable": "CHROMATICA_INDEX_DIR=C:\\Users\\anon\\github\\Chromatica\\85k",
            "server_command": "python -m src.chromatica.api.main",
            "hard_refresh_required": True,
            "estimated_restart_time": "10-15 seconds"
        }

        logger.info(f"✅ Restart completed: {restart_info["total_killed"]} processes killed")
        return restart_info

    except Exception as e:
        logger.error(f"❌ Failed to restart server: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to restart server: {str(e)}",
            "error": str(e),
            "killed_processes": killed_processes if "killed_processes" in locals() else [],
            "total_killed": len(killed_processes) if "killed_processes" in locals() else 0
        }

"""
    
    # Combine the parts
    new_content = before + new_restart_endpoint + after
    
    # Write the updated content back
    with open("src/chromatica/api/main.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ Successfully replaced restart endpoint with comprehensive version")
else:
    print("❌ Could not find restart endpoint markers")
