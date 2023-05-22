package ai.fedml.edge.utils;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Environment;
import android.os.StatFs;
import android.text.format.Formatter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import ai.fedml.edge.utils.entity.Memory;

public class MemoryUtils {

    public static Memory getMemory(Context context) {
        Memory memory = new Memory();
        try {
            memory.setRamMemoryTotal(getTotalMemory(context));
            memory.setRamMemoryAvailable(getMemoryAvailable(context));
            memory.setRomMemoryAvailable(getRomSpaceAvailable(context));
            memory.setRomMemoryTotal(getRomSpaceTotal(context));
        } catch (Exception e) {
            LogHelper.i(e.toString());
        }
        return memory;
    }

    private static String getTotalMemory(Context context) {
        String str1 = "/proc/meminfo";
        String str2;
        String[] arrayOfString;
        long initial_memory = 0;
        try {
            FileReader localFileReader;

            localFileReader = new FileReader(str1);

            BufferedReader localBufferedReader = new BufferedReader(localFileReader, 8192);
            str2 = localBufferedReader.readLine();

            arrayOfString = str2.split("\\s+");

            initial_memory = Long.valueOf(arrayOfString[1]) * 1024;
            localBufferedReader.close();

        } catch (Exception e) {
            LogHelper.i(e.toString());
        }
        return Formatter.formatFileSize(context, initial_memory);
    }

    private static String getMemoryAvailable(Context context) {
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        if (am != null) {
            am.getMemoryInfo(mi);
        }
        return Formatter.formatFileSize(context, mi.availMem);
    }

    private static String getRomSpaceAvailable(Context context) {
        File path = Environment.getDataDirectory();
        StatFs stat = new StatFs(path.getPath());
        long blockSize = stat.getBlockSize();
        long availableBlocks = stat.getAvailableBlocks();
        return Formatter.formatFileSize(context, availableBlocks * blockSize);
    }

    private static String getRomSpaceTotal(Context context) {
        File path = Environment.getDataDirectory();
        StatFs stat = new StatFs(path.getPath());
        long blockSize = stat.getBlockSize();
        long totalBlocks = stat.getBlockCount();
        return Formatter.formatFileSize(context, totalBlocks * blockSize);
    }

}