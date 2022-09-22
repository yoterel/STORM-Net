
#include "file_dialog.h"


bool getPath(nfdfilteritem_t filterItem[], string& path)
{
  nfdchar_t *outPath;
  nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 1, NULL);

  if (result == NFD_OKAY)
  {
    path = outPath;
    NFD_FreePath(outPath);
    return true;
  }
  return false;
}

bool getDirPath(string& path)
{
  nfdchar_t *outPath;
  nfdresult_t result = NFD_PickFolder(&outPath, NULL);

  if (result == NFD_OKAY)
  {
    path = outPath;
    NFD_FreePath(outPath);
    return true;
  }
  return false;
}

