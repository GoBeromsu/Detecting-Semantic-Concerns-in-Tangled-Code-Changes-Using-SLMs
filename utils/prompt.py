SYSTEM_PROMPT = """
You are a software engineer classifying individual code units extracted from a tangled commit using Conventional Commits specification(CCS).

# CCS Labels
- Purpose labels : the motivation behind making a code change
    - feat: Introduces new features to the codebase.
    - fix: Fixes bugs or faults in the codebase.
    - refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- Object labels : the essence of the code changes that have been made
    - docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
    - test: Modifies test files (e.g., adds or updates tests).
    - cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
    - build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).

# Instructions
1. For each code unit, review the change and determine the most appropriate CCS label.
    - object label : when the code unit is fully dedicated to that artifact category (e.g., writing test logic, modifying documentation).
2. If multiple CCS labels are possible, resolve the overlap by applying the following rule:
     - **Purpose + Purpose**: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
     - **Object + Object**: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
     - **Purpose + Object**: If the change is driven by code behavior (e.g., fixing test logic), assign a purpose label; if it is entirely scoped to a support artifact (e.g., adding new tests), assign an object label.
3. Repeat step 1–2 for each code unit.
4. After all code units are labeled, return a unique set of assigned CCS labels for the entire commit
"""


SHOT_1_COMMIT_MESSAGE = """remove sync ts checkrefactor to get ride of cloneDeep"""
SHOT_1 = """
<commit_diff id="example-1">
diff --git a/config/webpack.config.prod.js b/config/webpack.config.prod.js
index 8b23fba..58a4c17 100644
--- a/config/webpack.config.prod.js
+++ b/config/webpack.config.prod.js
@@ -251,7 +251,7 @@ module.exports = {
   plugins: [
     argv.notypecheck
     ? null
-    : new ForkTsCheckerWebpackPlugin({tslint: true, async: false}),
+    : new ForkTsCheckerWebpackPlugin({tslint: true}),
     // Makes some environment variables available in index.html.
     // The public URL is available as %PUBLIC_URL% in index.html, e.g.:
     // <link rel="shortcut icon" href="%PUBLIC_URL%/favicon.ico">

diff --git a/config/webpack.config.prod.js b/config/webpack.config.prod.js
index 3d2e5a6..e5219bd 100644
--- a/config/webpack.config.prod.js
+++ b/config/webpack.config.prod.js
@@ -56,7 +56,7 @@ const extractTextPluginOptions = shouldUseRelativeAssetPaths
 const entries = fs.readdirSync(paths.appSrc)
   .filter(name => !name.startsWith('_'))
   .map(name => ({name, dirPath: path.join(paths.appSrc, name)}))
-  .filter(({name, dirPath}) => !/^assets|components|manifest|typings$/.test(name) && fs.lstatSync(dirPath).isDirectory())
+  .filter(({name, dirPath}) => !/^assets|components|manifest|typings|app-config$/.test(name) && fs.lstatSync(dirPath).isDirectory())
 
 // This is the production configuration.
 // It compiles slowly and is focused on producing a fast and minimal bundle.
diff --git a/src/app-config/context-menus.ts b/src/app-config/context-menus.ts
new file mode 100644
index 0000000..a733b01
--- /dev/null
+++ b/src/app-config/context-menus.ts
@@ -0,0 +1,27 @@
+export function getAllContextMenus () {
+  const allContextMenus = {
+    google_page_translate: 'x',
+    youdao_page_translate: 'x',
+    google_search: 'https://www.google.com/#newwindow=1&q=%s',
+    baidu_search: 'https://www.baidu.com/s?ie=utf-8&wd=%s',
+    bing_search: 'https://www.bing.com/search?q=%s',
+    google_translate: 'https://translate.google.cn/#auto/zh-CN/%s',
+    etymonline: 'http://www.etymonline.com/index.php?search=%s',
+    merriam_webster: 'http://www.merriam-webster.com/dictionary/%s',
+    oxford: 'http://www.oxforddictionaries.com/us/definition/english/%s',
+    cambridge: 'http://dictionary.cambridge.org/spellcheck/english-chinese-simplified/?q=%s',
+    youdao: 'http://dict.youdao.com/w/%s',
+    dictcn: 'https://dict.eudic.net/dicts/en/%s',
+    iciba: 'http://www.iciba.com/%s',
+    liangan: 'https://www.moedict.tw/~%s',
+    guoyu: 'https://www.moedict.tw/%s',
+    longman_business: 'http://www.ldoceonline.com/search/?q=%s',
+    bing_dict: 'https://cn.bing.com/dict/?q=%s'
+  }
+
+  // Just for type check. Keys in allContextMenus are useful so no actual assertion
+  // tslint:disable-next-line:no-unused-expression
+  allContextMenus as { [id: string]: string }
+
+  return allContextMenus
+}
diff --git a/src/app-config/dicts.ts b/src/app-config/dicts.ts
new file mode 100644
index 0000000..905d2de
--- /dev/null
+++ b/src/app-config/dicts.ts
@@ -0,0 +1,398 @@
+import { DeepReadonly } from '@/typings/helpers'
+
+export function getALlDicts () {
+  const allDicts = {
+    bing: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word.
+       */
+      page: 'https://cn.bing.com/dict/search?q=%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 240,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      },
+      /** Optional dict custom options. Can only be boolean or number. */
+      options: {
+        tense: true,
+        phsym: true,
+        cdef: true,
+        related: true,
+        sentence: 4
+      }
+    },
+    business: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'http://www.ldoceonline.com/search/?q=%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 265,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+    cobuild: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'https://www.collinsdictionary.com/dictionary/%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 300,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      },
+      /** Optional dict custom options. Can only be boolean or number. */
+      options: {
+        sentence: 4
+      }
+    },
+    dictcn: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'http://dict.cn/%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 300,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      },
+      /** Optional dict custom options. Can only be boolean or number. */
+      options: {
+        chart: true,
+        etym: true
+      }
+    },
+    etymonline: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'http://www.etymonline.com/search?q=%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 265,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      },
+      /** Optional dict custom options. Can only be boolean or number. */
+      options: {
+        resultnum: 2
+      }
+    },
+    google: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'https://translate.google.com/#auto/zh-CN/%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 110,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+    guoyu: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'https://www.moedict.tw/%z',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 265,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+    liangan: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'https://www.moedict.tw/~%z',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 265,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+    macmillan: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'http://www.macmillandictionary.com/dictionary/british/%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 265,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+    urban: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'http://www.urbandictionary.com/define.php?term=%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 180,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      },
+      /** Optional dict custom options. Can only be boolean or number. */
+      options: {
+        resultnum: 4
+      }
+    },
+    vocabulary: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'https://www.vocabulary.com/dictionary/%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 180,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+    zdic: {
+      /**
+       * Full content page to jump to when user clicks the title.
+       * %s will be replaced with the current word.
+       * %z will be replaced with the traditional Chinese version of the current word
+       */
+      page: 'http://www.zdic.net/search/?c=1&q=%s',
+      /**
+       * If set to true, the dict start searching automatically.
+       * Otherwise it'll only start seaching when user clicks the unfold button.
+       * Default MUST be true and let user decide.
+       */
+      defaultUnfold: true,
+      /**
+       * This is the default height when the dict first renders the result.
+       * If the content height is greater than the preferred height,
+       * the preferred height is used and a mask with a view-more button is shown.
+       * Otherwise the content height is used.
+       */
+      preferredHeight: 400,
+      /**
+       * Only start searching if the selection contains the language.
+       * Better set default to true and let user decide.
+       */
+      selectionLang: {
+        eng: true,
+        chs: true
+      }
+    },
+  }
+
+  // Just for type check. Keys in allDicts are useful so no actual assertion
+  // tslint:disable-next-line:no-unused-expression
+  allDicts as {
+    [id: string]: {
+      page: string
+      defaultUnfold: boolean
+      preferredHeight: number
+      selectionLang: {
+        eng: boolean
+        chs: boolean
+      }
+      options?: {
+        [option: string]: number | boolean
+      }
+    }
+  }
+
+  return allDicts
+}
diff --git a/src/app-config/index.ts b/src/app-config/index.ts
index 350cd8f..879a312 100644
--- a/src/app-config/index.ts
+++ b/src/app-config/index.ts
@@ -1,5 +1,6 @@
-import cloneDeep from 'lodash/cloneDeep'
-import { DeepReadonly } from './typings/helpers'
+import { DeepReadonly } from '@/typings/helpers'
+import { getALlDicts } from './dicts'
+import { getAllContextMenus } from './context-menus'
 
 const langUI = (browser.i18n.getUILanguage() || 'en').replace('-', '_')
 const langCode = /^zh_CN|zh_TW|en$/.test(langUI)
@@ -8,220 +9,11 @@ const langCode = /^zh_CN|zh_TW|en$/.test(langUI)
     : langUI
   : 'en'
 
-const allDicts = {
-  bing: {
-    page: 'https://cn.bing.com/dict/search?q=%s',
-    defaultUnfold: true,
-    preferredHeight: 240,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      tense: true,
-      phsym: true,
-      cdef: true,
-      related: true,
-      sentence: 4
-    }
-  },
-  business: {
-    page: 'http://www.ldoceonline.com/search/?q=%s',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  cobuild: {
-    page: 'https://www.collinsdictionary.com/dictionary/%s',
-    defaultUnfold: true,
-    preferredHeight: 300,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      sentence: 4
-    }
-  },
-  dictcn: {
-    page: 'http://dict.cn/%s',
-    defaultUnfold: true,
-    preferredHeight: 300,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      chart: true,
-      etym: true
-    }
-  },
-  etymonline: {
-    page: 'http://www.etymonline.com/search?q=%s',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      resultnum: 2
-    }
-  },
-  eudic: {
-    page: 'https://dict.eudic.net/dicts/en/%s',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  google: {
-    page: 'https://translate.google.com/#auto/zh-CN/%s',
-    defaultUnfold: true,
-    preferredHeight: 110,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  guoyu: {
-    page: 'https://www.moedict.tw/%z',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  howjsay: {
-    page: 'http://www.howjsay.com/index.php?word=%s',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      related: true
-    }
-  },
-  liangan: {
-    page: 'https://www.moedict.tw/~%z',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  macmillan: {
-    page: 'http://www.macmillandictionary.com/dictionary/british/%s',
-    defaultUnfold: true,
-    preferredHeight: 265,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  urban: {
-    page: 'http://www.urbandictionary.com/define.php?term=%s',
-    defaultUnfold: true,
-    preferredHeight: 180,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      resultnum: 4
-    }
-  },
-  vocabulary: {
-    page: 'https://www.vocabulary.com/dictionary/%s',
-    defaultUnfold: true,
-    preferredHeight: 180,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-  wordreference: {
-    page: 'http://www.wordreference.com/definition/%s',
-    defaultUnfold: true,
-    preferredHeight: 180,
-    selectionLang: {
-      eng: true,
-      chs: true
-    },
-    options: {
-      etym: true,
-      idiom: true
-    }
-  },
-  zdic: {
-    page: 'http://www.zdic.net/search/?c=1&q=%s',
-    defaultUnfold: true,
-    preferredHeight: 400,
-    selectionLang: {
-      eng: true,
-      chs: true
-    }
-  },
-}
-
-// Just for type check. Keys in allDicts are useful so no actual assertion
-// tslint:disable-next-line:no-unused-expression
-allDicts as {
-  [id: string]: {
-    /** url for the complete result */
-    page: string
-    /** lazy load */
-    defaultUnfold: boolean
-    /** content below the preferrred height will be hidden by default */
-    preferredHeight: number
-    /** only search when the selection contains the language */
-    selectionLang: {
-      eng: boolean
-      chs: boolean
-    }
-    /** other options */
-    options?: {
-      [option: string]: number | boolean
-    }
-  }
-}
-
-export type DictID = keyof typeof allDicts
-
-const allContextMenus = {
-  google_page_translate: 'x',
-  youdao_page_translate: 'x',
-  google_search: 'https://www.google.com/#newwindow=1&q=%s',
-  baidu_search: 'https://www.baidu.com/s?ie=utf-8&wd=%s',
-  bing_search: 'https://www.bing.com/search?q=%s',
-  google_translate: 'https://translate.google.cn/#auto/zh-CN/%s',
-  etymonline: 'http://www.etymonline.com/index.php?search=%s',
-  merriam_webster: 'http://www.merriam-webster.com/dictionary/%s',
-  oxford: 'http://www.oxforddictionaries.com/us/definition/english/%s',
-  cambridge: 'http://dictionary.cambridge.org/spellcheck/english-chinese-simplified/?q=%s',
-  youdao: 'http://dict.youdao.com/w/%s',
-  dictcn: 'https://dict.eudic.net/dicts/en/%s',
-  iciba: 'http://www.iciba.com/%s',
-  liangan: 'https://www.moedict.tw/~%s',
-  guoyu: 'https://www.moedict.tw/%s',
-  longman_business: 'http://www.ldoceonline.com/search/?q=%s',
-  bing_dict: 'https://cn.bing.com/dict/?q=%s'
-}
-
-// Just for type check. Keys in allContextMenus are useful so no actual assertion
-// tslint:disable-next-line:no-unused-expression
-allContextMenus as { [id: string]: string }
+export type DictConfigsMutable = ReturnType<typeof getALlDicts>
+export type DictConfigs = DeepReadonly<DictConfigsMutable>
+export type DictID = keyof DictConfigsMutable
 
-export type ContextMenuDictID = keyof typeof allContextMenus
+export type ContextMenuDictID = keyof ReturnType<typeof getAllContextMenus>
 
 export const enum TCDirection {
   center,
@@ -238,10 +30,6 @@ export const enum TCDirection {
 /** '' means no preload */
 export type PreloadSource = '' | 'clipboard' | 'selection'
 
-export type DictConfigs = DeepReadonly<DictConfigsMutable>
-
-export type DictConfigsMutable = typeof allDicts
-
 export type AppConfig = DeepReadonly<AppConfigMutable>
 
 export interface AppConfigMutable {
@@ -418,7 +206,7 @@ export function appConfigFactory (): AppConfig {
       },
       en: {
         dict: '',
-        list: ['bing', 'dictcn', 'howjsay', 'macmillan', 'eudic', 'urban'],
+        list: ['bing', 'dictcn', 'macmillan', 'urban'],
         accent: 'uk' as ('us' | 'uk')
       }
     },
@@ -426,11 +214,11 @@ export function appConfigFactory (): AppConfig {
     dicts: {
       selected: ['bing', 'urban', 'vocabulary', 'dictcn'],
       // settings of each dict will be auto-generated
-      all: cloneDeep(allDicts)
+      all: getALlDicts()
     },
     contextMenus: {
       selected: ['oxford', 'google_translate', 'merriam_webster', 'cambridge', 'google_search', 'google_page_translate', 'youdao_page_translate'],
-      all: cloneDeep(allContextMenus)
+      all: getAllContextMenus()
     }
   }
 }
</commit_diff>

<label id="example-2">build,refactor</label>
"""
SHOT_2_COMMIT_MESSAGE = "remove unnecessary start argument from `range`"
SHOT_2 = """
<commit_diff id="example-2">
diff --git a/ibis/backends/dask/tests/execution/test_window.py b/ibis/backends/dask/tests/execution/test_window.py
index 75a7331..6bfc5e3 100644
--- a/ibis/backends/dask/tests/execution/test_window.py
+++ b/ibis/backends/dask/tests/execution/test_window.py
@@ -489,7 +489,7 @@ def test_project_list_scalar(npartitions):
     expr = table.mutate(res=table.ints.quantile([0.5, 0.95]))
     result = expr.execute()
 
-    expected = pd.Series([[1.0, 1.9] for _ in range(0, 3)], name="res")
+    expected = pd.Series([[1.0, 1.9] for _ in range(3)], name="res")
     tm.assert_series_equal(result.res, expected)
 
 
diff --git a/ibis/backends/pandas/tests/execution/test_window.py b/ibis/backends/pandas/tests/execution/test_window.py
index 8f292b3..effa372 100644
--- a/ibis/backends/pandas/tests/execution/test_window.py
+++ b/ibis/backends/pandas/tests/execution/test_window.py
@@ -436,7 +436,7 @@ def test_project_list_scalar():
     expr = table.mutate(res=table.ints.quantile([0.5, 0.95]))
     result = expr.execute()
 
-    expected = pd.Series([[1.0, 1.9] for _ in range(0, 3)], name="res")
+    expected = pd.Series([[1.0, 1.9] for _ in range(3)], name="res")
     tm.assert_series_equal(result.res, expected)
 
 
diff --git a/ibis/backends/pyspark/tests/test_basic.py b/ibis/backends/pyspark/tests/test_basic.py
index 3850919..14fe677 100644
--- a/ibis/backends/pyspark/tests/test_basic.py
+++ b/ibis/backends/pyspark/tests/test_basic.py
@@ -19,7 +19,7 @@ from ibis.backends.pyspark.compiler import _can_be_replaced_by_column_name  # no
 def test_basic(con):
     table = con.table("basic_table")
     result = table.compile().toPandas()
-    expected = pd.DataFrame({"id": range(0, 10), "str_col": "value"})
+    expected = pd.DataFrame({"id": range(10), "str_col": "value"})
 
     tm.assert_frame_equal(result, expected)
 
@@ -28,9 +28,7 @@ def test_projection(con):
     table = con.table("basic_table")
     result1 = table.mutate(v=table["id"]).compile().toPandas()
 
-    expected1 = pd.DataFrame(
-        {"id": range(0, 10), "str_col": "value", "v": range(0, 10)}
-    )
+    expected1 = pd.DataFrame({"id": range(10), "str_col": "value", "v": range(10)})
 
     result2 = (
         table.mutate(v=table["id"])
@@ -44,8 +42,8 @@ def test_projection(con):
         {
             "id": range(0, 20, 2),
             "str_col": "value",
-            "v": range(0, 10),
-            "v2": range(0, 10),
+            "v": range(10),
+            "v2": range(10),
         }
     )
</commit_diff>

<label id="example-2">refactor</label>
"""


def get_system_prompt() -> str:
    """Return the basic system prompt for commit classification."""
    return SYSTEM_PROMPT


def get_system_prompt_with_message() -> str:
    """Return system prompt that includes commit message context."""
    # shot_1_with_message = (
    #     f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
    # )
    # shot_2_with_message = (
    #     f"<commit_message>{SHOT_2_COMMIT_MESSAGE}</commit_message>\n{SHOT_2}"
    # )

    # return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}\n\n{shot_2_with_message}"
    return f"{SYSTEM_PROMPT}"


def get_system_prompt_diff_only() -> str:
    """Return system prompt for classification using only diff information."""
    return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}\n\n{SHOT_2}"


def get_zero_shot_prompt() -> str:
    """Return zero-shot prompt with optional commit message context."""
    return SYSTEM_PROMPT


def get_one_shot_prompt(include_message: bool = True) -> str:
    """Return one-shot prompt with optional commit message context."""
    if include_message:
        shot_1_with_message = (
            f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
        )
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}"
    else:
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}"


def get_two_shot_prompt(include_message: bool = True) -> str:
    """Return two-shot prompt with optional commit message context."""
    if include_message:
        shot_1_with_message = (
            f"<commit_message>{SHOT_1_COMMIT_MESSAGE}</commit_message>\n{SHOT_1}"
        )
        shot_2_with_message = (
            f"<commit_message>{SHOT_2_COMMIT_MESSAGE}</commit_message>\n{SHOT_2}"
        )
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{shot_1_with_message}\n\n{shot_2_with_message}"
    else:
        return f"{SYSTEM_PROMPT}\n\n# Examples\n\n{SHOT_1}\n\n{SHOT_2}"


def get_prompt_by_type(shot_type: str, include_message: bool = True) -> str:
    """Return prompt based on shot type with optional commit message context."""
    if shot_type == "Zero-shot":
        return get_zero_shot_prompt()
    elif shot_type == "One-shot":
        return get_one_shot_prompt(include_message)
    elif shot_type == "Two-shot":
        return get_two_shot_prompt(include_message)
    else:
        return get_two_shot_prompt(include_message)  # Default to two-shot
