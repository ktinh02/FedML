import subprocess
from datetime import datetime, timezone, timedelta
import json

class CriClient:
    def __init__(self):
        # self.runtime_endpoint = "unix:///run/containerd/containerd.sock"
        pass
    
    def _run_command(self, cmd):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            return None
    
    def get_container_id(self, name):
        cmd = ["crictl", "ps", "-q", "--name", name]
        return self._run_command(cmd).strip()
    
    def get_logs(self, container_id, since=None, follow=False):
        cmd = ["crictl", "logs"]
        
        if follow:
            cmd.append("-f")
        
        if since:
            cmd.extend(["--since", since])
            
        cmd.append(container_id)
        return self._run_command(cmd)
    
    def inspect_container(self, container_id):
        cmd = ["crictl", "inspect", container_id]
        output = self._run_command(cmd)
        if output:
            return json.loads(output)
        return None

# 使用示例
if __name__ == "__main__":
    client = CriClient()
    
    # 获取容器ID
    container_id = client.get_container_id("app-container")
    if container_id:
        print(f"Container ID: {container_id}")
        
        # 获取容器信息
        info = client.inspect_container(container_id)
        if info:
            print("Container info:", json.dumps(info, indent=2))
        
        # 获取最近2分钟的日志
        two_mins_ago = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        logs = client.get_logs(container_id, since=two_mins_ago)
        print("Recent logs:", logs)