using Whisper.net;
using Whisper.net.Ggml;
using Whisper.net.Logger;

var ggmlType = GgmlType.Small;
var modelFileName = "ggml-small.bin";
var wavFilePathName = "wav file path";

using var whisperLogger = LogProvider.AddConsoleLogging(WhisperLogLevel.Debug);

if (!File.Exists(modelFileName))
{
    await DownloadModel(modelFileName, ggmlType);
}

using var whisperFactory = WhisperFactory.FromPath("ggml-small.bin");

using var processor = whisperFactory.CreateBuilder()
    .WithLanguage("auto")
    .Build();

using var fileStream = File.OpenRead(wavFilePathName);

await foreach (var result in processor.ProcessAsync(fileStream))
{
    Console.WriteLine($"{result.Start}->{result.End}: {result.Text}");
}

static async Task DownloadModel(string fileName, GgmlType ggmlType)
{
    Console.WriteLine($"Downloading Model {fileName}");
    using var modelStream = await WhisperGgmlDownloader.Default.GetGgmlModelAsync(ggmlType);
    using var fileWriter = File.OpenWrite(fileName);
    await modelStream.CopyToAsync(fileWriter);
}