using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class Launcher {
    static string Str(Dictionary<string, object> d, string k) { return d.ContainsKey(k) && d[k] != null ? (string)d[k] : null; }
    static string Hash(string p) { using (var s=File.OpenRead(p)) using(var h=SHA256.Create()) return BitConverter.ToString(h.ComputeHash(s)).Replace("-", "").ToLowerInvariant(); }
    static string Quote(string s) {
        var b=new StringBuilder("\""); int n=0;
        foreach(char c in s) { if(c=='\\') { n++; continue; } if(c=='\"') { b.Append('\\',n*2+1).Append(c); n=0; continue; } b.Append('\\',n).Append(c); n=0; }
        return b.Append('\\',n*2).Append('"').ToString();
    }
    static void Within(string root, string child) {
        string r=Path.GetFullPath(root).TrimEnd('\\')+"\\", p=Path.GetFullPath(child);
        if(!p.StartsWith(r,StringComparison.OrdinalIgnoreCase)) throw new Exception("Component path escaped its directory.");
    }
    static async Task Pump(Stream input, Stream output) {
        var bytes=new byte[8192]; int count;
        while((count=await input.ReadAsync(bytes,0,bytes.Length))>0) {
            await output.WriteAsync(bytes,0,count);
            // MCP is interactive: never wait for EOF or a full buffered stream.
            await output.FlushAsync();
        }
    }
    [STAThread] static int Main(string[] args) {
#if MCP
        bool mcp=true;
#else
        bool mcp=false;
#endif
        FileStream runtimeLease=null;
        try {
            string home=AppDomain.CurrentDomain.BaseDirectory;
            string gate=Path.Combine(home,"runtime-use.lock");
            if(!File.Exists(gate) || (File.GetAttributes(gate)&FileAttributes.ReparsePoint)!=0)
                throw new Exception("安装使用记录缺失或无效，请修复 Aester 安装。");
            // Hold the same read/shared-read lease as every Python process,
            // starting before state reads and lasting through child shutdown.
            runtimeLease=new FileStream(gate,FileMode.Open,FileAccess.Read,FileShare.Read);
            if(File.Exists(Path.Combine(home,"uninstall.pending")))
                throw new Exception("Aester 正在维护或上次维护未完成，请等待安装完成、重试卸载或修复安装。");
            var json=new JavaScriptSerializer();
            var state=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(Path.Combine(home,"install-state.json"),Encoding.UTF8));
            string root=Str(state,"component_root"), rp=Str(state,"receipt_path"), expectedHash=Str(state,"receipt_sha256");
            string currentPath=Path.Combine(root,"current.json");
            if(File.Exists(currentPath)) {
                var pointer=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(currentPath,Encoding.UTF8));
                string version=Str(pointer,"version");
                if(String.IsNullOrEmpty(version)||!System.Text.RegularExpressions.Regex.IsMatch(version,@"\A[A-Za-z0-9-]+\z")) throw new Exception("Invalid component version pointer.");
                string candidate=Path.Combine(root,"versions",version,"receipt.json"); Within(root,candidate);
                if(Hash(candidate)!=Str(pointer,"receipt_sha256")) throw new Exception("Current component receipt changed.");
                var current=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(candidate,Encoding.UTF8));
                // A GUI optional-component upgrade may switch only this code version.
                if(String.Equals(Str(current,"app_root"),Str(state,"app_root"),StringComparison.OrdinalIgnoreCase)) {
                    rp=candidate; expectedHash=Str(pointer,"receipt_sha256");
                }
            }
            Within(root,rp);
            if(Hash(rp)!=expectedHash) throw new Exception("Component receipt changed. Please repair the installation.");
            var receipt=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(rp,Encoding.UTF8));
            if(!String.Equals(Str(receipt,"app_root"),Str(state,"app_root"),StringComparison.OrdinalIgnoreCase)) throw new Exception("Component belongs to another application version.");
            string python=Str(receipt,mcp?"python":"pythonw"), siril=Str(receipt,"siril"), app=Str(state,"app_root");
            Within(root,python); Within(root,siril); Within(home,app);
            if(!File.Exists(python)||!File.Exists(siril)||!Directory.Exists(app)) throw new Exception("Installation is incomplete. Please run Setup again.");
            // The entry module comes from the application's own PACKAGE_CONTENTS.json ("entry_modules"); application
            // versions built before the open/closed split have none and keep their original entry points.
            string entry=mcp?"starstack.mcp_server":"starstack.gui";
            string contentsPath=Path.Combine(app,"PACKAGE_CONTENTS.json");
            if(File.Exists(contentsPath)) {
                var contents=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(contentsPath,Encoding.UTF8));
                var modules=contents.ContainsKey("entry_modules")?contents["entry_modules"] as Dictionary<string,object>:null;
                string chosen=modules!=null?Str(modules,mcp?"mcp":"gui"):null;
                if(chosen!=null) {
                    if(!System.Text.RegularExpressions.Regex.IsMatch(chosen,@"\A[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\z"))
                        throw new Exception("Invalid application entry module. Please repair the installation.");
                    entry=chosen;
                }
            }
            var start=new ProcessStartInfo(python,"-B -X utf8 -m "+entry);
            start.UseShellExecute=false; start.CreateNoWindow=true; start.WorkingDirectory=app;
            foreach(string a in args) start.Arguments+=" "+Quote(a);
            start.EnvironmentVariables["AESTER_COMPONENT_ROOT"]=root;
            start.EnvironmentVariables["AESTER_APP_ROOT"]=app;
            start.EnvironmentVariables["AESTER_INSTALL_ROOT"]=home.TrimEnd('\\');
            start.EnvironmentVariables["STARSTACK_SIRIL"]=siril;
            string model=Str(receipt,"model_path");
            if(model!=null) { Within(root,model); start.EnvironmentVariables["STARSTACK_STAR_MODEL"]=model; start.EnvironmentVariables["STARSTACK_STAR_ADAPTER"]=json.Serialize(receipt["star_backend_options"]); }
            if(mcp) {
                // A no-console child does not reliably inherit pipe handles on Windows.
                // Forward the three byte streams explicitly; never print protocol banners.
                start.RedirectStandardInput=true; start.RedirectStandardOutput=true; start.RedirectStandardError=true;
                using(var child=Process.Start(start)) {
                    var input=Pump(Console.OpenStandardInput(),child.StandardInput.BaseStream).ContinueWith(t=>{try{child.StandardInput.Close();}catch{}});
                    var output=Pump(child.StandardOutput.BaseStream,Console.OpenStandardOutput());
                    var error=Pump(child.StandardError.BaseStream,Console.OpenStandardError());
                    child.WaitForExit(); Task.WaitAll(output,error); return child.ExitCode;
                }
            }
            Directory.CreateDirectory(Path.Combine(root,"logs"));
            using(var log=new StreamWriter(Path.Combine(root,"logs","gui-"+DateTime.Now.ToString("yyyyMMdd-HHmmss-fff")+".log"),false,Encoding.UTF8)) {
                log.AutoFlush=true; start.RedirectStandardOutput=true; start.RedirectStandardError=true;
                using(var child=new Process()) {
                    child.StartInfo=start;
                    child.OutputDataReceived+=(s,e)=>{ if(e.Data!=null) lock(log) log.WriteLine(e.Data); };
                    child.ErrorDataReceived+=(s,e)=>{ if(e.Data!=null) lock(log) log.WriteLine(e.Data); };
                    child.Start(); child.BeginOutputReadLine(); child.BeginErrorReadLine(); child.WaitForExit();
                    if(child.ExitCode!=0) throw new Exception("Aester stopped. The diagnostic log is in "+Path.Combine(root,"logs"));
                }
            }
            return 0;
        } catch(Exception e) {
            if(mcp) Console.Error.WriteLine("Aester: "+e.Message);
            else MessageBox.Show(e.Message,"Aester",MessageBoxButtons.OK,MessageBoxIcon.Error);
            return 1;
        } finally {
            if(runtimeLease!=null) runtimeLease.Dispose();
        }
    }
}
