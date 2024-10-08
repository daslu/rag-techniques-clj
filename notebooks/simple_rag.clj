(ns simple-rag
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [pdfboxing.text]
            [libpython-clj2.python :as py :refer [py.]]
            [libpython-clj2.require :refer [require-python]]
            [wkok.openai-clojure.api :as openai]
            [ragtacts.splitter.recursive]
            [ragtacts.splitter.base]
            [ragtacts.embedding.base]
            [ragtacts.core :as ragtacts])
  (:import (dev.langchain4j.data.document Document)
           (dev.langchain4j.data.document.splitter DocumentSplitters)))

(require-python '[langchain_text_splitters]
                '[builtins])

;; Inspired by: https://github.com/NirDiamant/RAG_Techniques/blob/004aabf/all_rag_techniques/simple_rag.ipynb

(def pdf-path "data/Understanding_Climate_Change.pdf")

(-> pdf-path
    pdfboxing.text/extract
    (subs 0 4000)
    ragtacts.embedding.base/text->doc
    (->> (ragtacts.splitter.base/split
          (ragtacts.splitter.recursive/recursive-splitter
           {:size 500
            :overlap 200}))))

(let [db (ragtacts/vector-store)]
  (ragtacts/add db [(pdfboxing.text/extract pdf-path)])
  (ragtacts/search db "Tell me about Climate."))





(->> pdf-path
     pdfboxing.text/extract
     ragtacts.embedding.base/text->doc
     )


(.split (DocumentSplitters/recursive 1000 200)
        (Document. (-> pdf-path
                       (str/replace #"\t" " ")
                       pdfboxing.text/extract)))

(defn ->embedding [chunk]
  (openai/create-embedding {:model "text-embedding-3-large"
                            :input chunk}))

(defn encode-pdf [{:keys [chunk-size chunk-overlap]
                   :or {chunk-size 1000
                        chunk-overlap 200}}]
  (->> pdf-path
       pdfboxing.text/extract
       (Document.)
       (.split (DocumentSplitters/recursive chunk-size
                                            chunk-overlap))
       ;; (map (fn [chunk]
       ;;        (-> chunk
       ;;            (str/replace #"\t" " ")
       ;;            ;; ->embedding
       ;;            )))
       ))

(->> (encode-pdf {})
     )









(ragtacts.core/c)
